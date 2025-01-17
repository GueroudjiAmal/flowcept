from typing import List, Union

from flowcept.commons.flowcept_dataclasses.workflow_object import (
    WorkflowObject,
)

from uuid import uuid4

import flowcept.instrumentation.decorators
from flowcept.commons import logger
from flowcept.commons.daos.document_db_dao import DocumentDBDao
from flowcept.commons.daos.mq_dao.mq_dao_base import MQDao
from flowcept.configs import (
    MQ_INSTANCES,
    INSTRUMENTATION_ENABLED,
)
from flowcept.flowcept_api.db_api import DBAPI
from flowcept.commons.flowcept_logger import FlowceptLogger
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor

class Flowcept(object):
    db = DBAPI()

    current_workflow_id = None

    def __init__(
            self,
            interceptors: Union[BaseInterceptor, List[BaseInterceptor], str] = None,
            bundle_exec_id=None,
            campaign_id: str = None,
            workflow_id: str = None,
            workflow_name: str = None,
            workflow_args: str = None,
            start_persistence=True,
            save_workflow=True,
    ):
        """Flowcept controller.

        This class controls the interceptors, including instrumentation.
        If using for instrumentation, we assume one instance of this class
        per workflow is being utilized.

        Parameters
        ----------
        interceptors - list of Flowcept interceptors. If none, instrumentation
        will be used. If a string is passed, no interceptor will be
        started. # TODO: improve clarity for the documentation.

        bundle_exec_id - A way to group interceptors.

        start_persistence - Whether you want to persist the messages in one of the DBs defined in
         the `databases` settings.
         save_workflow - Whether you want to send a workflow object message.
        """
        self.logger = FlowceptLogger()
        self._enable_persistence = start_persistence
        self._db_inserters: List = []
        if bundle_exec_id is None:
            self._bundle_exec_id = id(self)
        else:
            self._bundle_exec_id = bundle_exec_id
        self.enabled = True
        self.is_started = False
        if isinstance(interceptors, str):
            self._interceptors = None
        else:
            if interceptors is None:
                if not INSTRUMENTATION_ENABLED:
                    self.enabled = False
                    return
                interceptors = [
                    flowcept.instrumentation.decorators.instrumentation_interceptor
                ]
            elif not isinstance(interceptors, list):
                interceptors = [interceptors]
            self._interceptors: List[BaseInterceptor] = interceptors

        self._save_workflow = save_workflow
        self.current_workflow_id = workflow_id
        self.campaign_id = campaign_id
        self.workflow_name = workflow_name
        self.workflow_args = workflow_args

    def start(self):
        """Start it."""
        if self.is_started or not self.enabled:
            self.logger.warning("DB inserter may be already started or instrumentation is not set")
            return self

        if self._enable_persistence:
            self.logger.debug("Flowcept persistence starting...")
            if MQ_INSTANCES is not None and len(MQ_INSTANCES):
                for mq_host_port in MQ_INSTANCES:
                    split = mq_host_port.split(":")
                    mq_host = split[0]
                    mq_port = int(split[1])
                    self._init_persistence(mq_host, mq_port)
            else:
                self._init_persistence()

        if self._interceptors and len(self._interceptors):
            for interceptor in self._interceptors:
                # TODO: :base-interceptor-refactor: revise
                if interceptor.settings is None:
                    key = id(interceptor)
                else:
                    key = interceptor.settings.key
                self.logger.debug(f"Flowceptor {key} starting...")
                interceptor.start(bundle_exec_id=self._bundle_exec_id)
                self.logger.debug(f"...Flowceptor {key} started ok!")

                if interceptor.kind == "instrumentation":
                    Flowcept.current_workflow_id = self.current_workflow_id or str(uuid4())
                    Flowcept.campaign_id = self.campaign_id or str(uuid4())
                    if self._save_workflow:
                        wf_obj = WorkflowObject()
                        wf_obj.workflow_id = Flowcept.current_workflow_id
                        wf_obj.campaign_id = Flowcept.campaign_id
                        if self.workflow_name:
                            wf_obj.name = self.workflow_name
                        if self.workflow_args:
                            wf_obj.used = self.workflow_args
                        interceptor.send_workflow_message(wf_obj)
                else:
                    Flowcept.current_workflow_id = None

        self.logger.debug("Ok, we're consuming messages to persist!")
        self.is_started = True
        return self

    def _init_persistence(self, mq_host=None, mq_port=None):
        from flowcept.flowceptor.consumers.document_inserter import DocumentInserter

        self._db_inserters.append(
            DocumentInserter(
                check_safe_stops=True,
                bundle_exec_id=self._bundle_exec_id,
                mq_host=mq_host,
                mq_port=mq_port,
            ).start()
        )
        print("We started doc inserter!")

    def stop(self):
        """Stop it."""
        print("Going to stop everything!")
        if not self.is_started or not self.enabled:
            self.logger.warning("Flowcept is already stopped or may never have been started!")
            return
        if self._interceptors and len(self._interceptors):
            for interceptor in self._interceptors:
                # TODO: :base-interceptor-refactor: revise
                if interceptor.settings is None:
                    key = id(interceptor)
                else:
                    key = interceptor.settings.key
                self.logger.info(f"Flowceptor {key} stopping...")
                interceptor.stop()

        print("BLA")
        from time import sleep
        sleep(0.5)

        if len(self._db_inserters):
            self.logger.info("Stopping DB Inserters...")
            for db_inserter in self._db_inserters:
                db_inserter.stop(bundle_exec_id=self._bundle_exec_id)
        self.is_started = False
        self.logger.debug("All stopped!")

    def __enter__(self):
        """Run the start function."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run the stop function."""
        self.stop()

    @staticmethod
    def services_alive() -> bool:
        if not MQDao.build().liveness_test():
            logger.error("MQ Not Ready!")
            return False
        if not DocumentDBDao().liveness_test():
            logger.error("DocDB Not Ready!")
            return False
        logger.info("MQ and DocDB are alive!")
        return True
