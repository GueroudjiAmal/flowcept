import sys
import os
import time
from threading import Thread, Event

from watchdog.observers import Observer

from flowcept.commons.flowcept_data_classes import TaskMessage
from flowcept.commons.utils import get_utc_now, get_status_from_str
from flowcept.flowceptor.plugins.base_interceptor import (
    BaseInterceptor,
)
from flowcept.flowceptor.plugins.interceptor_state_manager import (
    InterceptorStateManager,
)

from flowcept.flowceptor.plugins.mlflow.mlflow_dao import MLFlowDAO
from flowcept.flowceptor.plugins.mlflow.interception_event_handler import (
    InterceptionEventHandler,
)
from flowcept.flowceptor.plugins.mlflow.mlflow_dataclasses import RunData


class MLFlowInterceptor(BaseInterceptor):
    def __init__(self, plugin_key="mlflow"):
        super().__init__(plugin_key)
        self._observer_thread = None
        self.state_manager = InterceptorStateManager(self.settings)
        self.dao = MLFlowDAO(self.settings)

    def prepare_task_msg(self, mlflow_run_data: RunData) -> TaskMessage:
        task_msg = TaskMessage()
        task_msg.task_id = mlflow_run_data.task_id
        task_msg.utc_timestamp = get_utc_now()
        task_msg.status = get_status_from_str(mlflow_run_data.status)
        task_msg.used = mlflow_run_data.used
        task_msg.generated = mlflow_run_data.generated
        return task_msg

    def callback(self):
        """
        This function is called whenever a change is identified in the data.
        It decides what to do in the event of a change.
        If it's an interesting change, it calls self.intercept; otherwise,
        let it go....
        """
        runs = self.dao.get_finished_run_uuids()
        for run_uuid_tuple in runs:
            run_uuid = run_uuid_tuple[0]
            if not self.state_manager.has_element_id(run_uuid):
                self.logger.debug(
                    f"We need to intercept this Run: {run_uuid}"
                )
                run_data = self.dao.get_run_data(run_uuid)
                self.state_manager.add_element_id(run_uuid)
                task_msg = self.prepare_task_msg(run_data)
                self.intercept(task_msg)

    def start(self):
        self.observe()

    def stop(self):
        self.logger.debug("Interceptor stopping...")
        self._observer.stop()
        self.logger.debug("Interceptor stopped.")

    def observe(self):
        event_handler = InterceptionEventHandler(
            self, self.__class__.callback
        )
        while not os.path.isfile(self.settings.file_path):
            self.logger.warning(
                f"I can't watch the file {self.settings.file_path},"
                f" as it does not exist."
                f"\tI will sleep for {self.settings.watch_interval_sec} sec."
                f" to see if it appears."
            )
            time.sleep(self.settings.watch_interval_sec)

        self._observer = Observer()
        self._observer.schedule(
            event_handler, self.settings.file_path, recursive=True
        )
        self._observer.start()
        self.logger.info(f"Watching {self.settings.file_path}")


if __name__ == "__main__":
    try:
        interceptor = MLFlowInterceptor()
        interceptor.observe()
        while True:
            time.sleep(interceptor.settings.watch_interval_sec)
    except KeyboardInterrupt:
        sys.exit(0)
