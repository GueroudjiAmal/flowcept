import sys

import yaml

from flowcept import (
    FlowceptConsumerAPI,
    ZambezeInterceptor,
    MLFlowInterceptor,
    TensorboardInterceptor,
)
from flowcept.commons.vocabulary import Vocabulary
from flowcept.configs import SETTINGS_PATH, EMBEDDED_OBSERVERS


INTERCEPTORS = {
    Vocabulary.Settings.ZAMBEZE_KIND: ZambezeInterceptor,
    Vocabulary.Settings.MLFLOW_KIND: MLFlowInterceptor,
    Vocabulary.Settings.TENSORBOARD_KIND: TensorboardInterceptor,
    # Vocabulary.Settings.DASK_KIND: DaskInterceptor,
}


def main():
    # TODO: this is unfinished
    with open(SETTINGS_PATH) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    for plugin_key in yaml_data["plugins"]:
        plugin_settings_obj = yaml_data["plugins"][plugin_key]
        if (
            "enabled" in plugin_settings_obj
            and not plugin_settings_obj["enabled"]
        ):
            continue

        kind = plugin_settings_obj["kind"]

        interceptor = None
        if kind in INTERCEPTORS:
            interceptor = INTERCEPTORS[plugin_settings_obj["kind"]](
                plugin_key
            )
        consumer = FlowceptConsumerAPI(interceptor)
        consumer.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
