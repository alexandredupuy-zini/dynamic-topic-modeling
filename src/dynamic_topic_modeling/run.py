"""Application entry point."""

from pathlib import Path
from typing import Iterable, Type, Dict

from kedro.context import KedroContext, load_context
from kedro.runner import AbstractRunner
from kedro.pipeline import Pipeline

from dynamic_topic_modeling.pipeline import create_pipelines


class ProjectContext(KedroContext):
    """Users can override the remaining methods from the parent class here, or create new ones
    (e.g. as required by plugins)

    """

    project_name = "Dynamic Topic Modeling"
    project_version = "0.15.4"

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines()


def main(
    tags: Iterable[str] = None,
    env: str = None,
    runner: Type[AbstractRunner] = None,
    node_names: Iterable[str] = None,
    from_nodes: Iterable[str] = None,
    to_nodes: Iterable[str] = None,
    from_inputs: Iterable[str] = None,
):
    """Application main entry point.

    Args:
        tags: An optional list of node tags which should be used to
            filter the nodes of the ``Pipeline``. If specified, only the nodes
            containing *any* of these tags will be run.
        env: An optional parameter specifying the environment in which
            the ``Pipeline`` should be run.
        runner: An optional parameter specifying the runner that you want to run
            the pipeline with.
        node_names: An optional list of node names which should be used to filter
            the nodes of the ``Pipeline``. If specified, only the nodes with these
            names will be run.
        from_nodes: An optional list of node names which should be used as a
            starting point of the new ``Pipeline``.
        to_nodes: An optional list of node names which should be used as an
            end point of the new ``Pipeline``.
        from_inputs: An optional list of input datasets which should be used as a
            starting point of the new ``Pipeline``.

    """
    project_context = load_context(Path.cwd(), env=env)
    project_context.run(
        tags=tags,
        runner=runner,
        node_names=node_names,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        from_inputs=from_inputs,
    )


if __name__ == "__main__":
    main()
