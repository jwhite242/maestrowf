"""
Prototype indexer for maestro studies.  Given a maestro study workspace, this
will reconstruct the metadata/dag and iterate over the step workspaces and
the values of any maestro parameters associated with them.

Todo
----
-[ ] Add study root discovery: -> scan upwards, looking for meta folder
     and verify parameters.yaml and metadata.yaml are in them - reduce
     need to also pass in study root dir
-[X] Add attributes/namedtuple for the iterator output to be more self_documenting
-[ ] Add better handler for steps without param combo strings
-[ ] Improve parsing/typing of status data with data classes
-[ ] Add dfs/bfs iteration options -> pack up the data in a pyaestro dag?
-[X] Add a dot file output option with styles/text attribute inclusion filters
  - [ ] Add node style maps: node shape/color/some other highlight based on state
  - [ ] Per step name node styles
  - [ ] Add subgraphs to highlight all nodes with same param combos?

Authors
-------
  Jessica Semler
  Jeremy White
"""
import argparse
import os
import sys

from attrs import define, field, Factory
from maestrowf.utils import csvtable_to_dict
from pyaestro.structures.graphs import AcyclicAdjGraph
from pyaestro.structures.graphs.algorithms import DepthFirstSearch, BreadthFirstSearch

from collections import namedtuple
from itertools import cycle
from rich.pretty import pprint
from rich.console import Console

import yaml

console = Console()

StepInfo = namedtuple('StepInfo',
                      ['name', 'workspace_name', 'workspace_path', 'parameters', 'status'])

SAMPLE_THEME = {
    "hello_world": {
        "shape": "ellipse",
        "fill": "#4F378B",
        "fontcolor": "#FFFBFE"
    },
    "bye_world": {
        "shape": "ellipse",
        "fill": "#633B48",
        "fontcolor": "#FFFBFE"
    },
    "_source": {
        "shape": "ellipse",
        "fill": "#49454F",
        "fontcolor": "#FFFBFE"
    },
    "default": {
        "shape": "ellipse",
        "fill": "#8C1D18",
        "fontcolor": "#FFFBFE"
    },
}

NODE_STYLE_TEMPLATE = '{0} [label="{1}" fontcolor="{2[fontcolor]}" color="{2[fill]}" style="filled"]'

def make_dot_node_str(node_name, node_class, node_str, theme=SAMPLE_THEME, node_style_template=NODE_STYLE_TEMPLATE):
    if node_class not in theme:
        node_theme = theme['default']
    else:
        node_theme = theme[node_class]
    pprint(node_theme)
    # return '{0} [label="{1}" fontcolor={2["fontcolor"]} color={2["fill"]} style="filled"]'.format(node_name, node_str, node_theme)
    return NODE_STYLE_TEMPLATE.format(node_name, node_str, node_theme)


def load_yaml(path):
    with open(path, 'r') as _file:
        return yaml.load(_file, Loader=yaml.Loader)


class MaestroStudyMixin(object):
    """This mixin provides an interface to a Maestro run object."""

    @property
    def status(self):
        """
        The Maestro status.

        :returns: A ``MaestroStatus`` object representing the ``status.csv``
            file.
        """
        if os.path.isfile(self.status_csv):
            status_file = csvtable_to_dict(open(self.status_csv))
            return MaestroStatus.from_status_csv(status_file)
            # return "STATUS OBJECT"  # MaestroStatus(self.status_csv)

    @property
    def status_csv(self):
        """
        The path to the Maestro status file.

        :returns: The path to the Maestro ``status.csv`` file.
        """
        return os.path.join(self.path, 'status.csv')

    @property
    def logs_dir(self):
        """
        The Maestro log directory path.

        :returns: The path to the Maestro logs directory.
        """
        return os.path.join(self.path, 'logs')

    @property
    def meta_dir(self):
        """
        The Maestro meta directory path.

        :returns: The path to the Maestro meta directory.
        """
        return os.path.join(self.path, 'meta')

    @property
    def is_valid(self):
        """Tests whether this is a valid Maestro study by testing metadata files."""

        # Be good to check status as well to verify the study has actually been run?
        if os.path.exists(os.path.join(self.meta_dir, 'metadata.yaml')):
            return True
        else:
            return False

    @property
    def meta(self):
        return Meta(self.meta_dir)

    def __iter__(self):
        """Iterator over workspaces, yielding StepInfo namedtuples"""
        for workspace_name, workspace_path in self.meta._get_workspaces():
            step, params, _ = self._find_param(workspace_path)
            step_status = self.status.row_by_workspace(workspace_path)
            yield StepInfo(step, workspace_name, workspace_path, params, step_status)

    def _find_param(self, workspace_path):
        """Searches for workspace_path to get step name and parameters"""
        for step, stripped_combo, step_workspace in self.meta._get_steps():
            if workspace_path == step_workspace:
                if stripped_combo in self.meta.parameter_map:
                    params = self.meta.parameters[
                        self.meta.parameter_map[stripped_combo]]
                else:
                    params = None  # More useful to have an empty iterable downstream?

                return step, params, stripped_combo

    @property
    def study_graph(self):
        return self._build_graph()

    def _build_graph(self):
        study_graph = AcyclicAdjGraph()
        study_graph['_source'] = MaestroStep(step_name='root',
                                             workspace_name='root',
                                             workspace_path='$(OUTPUT_PATH)',
                                             status='NONE',
                                             param_combo_str='',
                                             param_list=[])

        graph_steps = []        # store list of steps for later use in dependency linking to avoid extra work
        # Add all the nodes
        for step_info in self.__iter__():
            step, params, stripped_combo = self._find_param(step_info.workspace_path)
            if params:
                param_dict = params['params']
            else:
                param_dict = {}

            study_graph[(step_info.name, step_info.workspace_name)] = MaestroStep.create_step(
                step_name=step_info.name,
                workspace_name=step_info.workspace_name,
                workspace_path=step_info.workspace_path,
                status='NONE',
                param_combo_str=stripped_combo,
                param_dict=param_dict
            )
            # pprint("Adding Node")
            # pprint(study_graph[(step_info.name, step_info.workspace_name)])
            
            graph_steps.append(study_graph[(step_info.name, step_info.workspace_name)])

        # Start adding the edges using dependencies, starting by adding dependency free edges to _source
        visited_dependencies = {}
        raw_dependencies = {step_name: step_dependencies for step_name, step_dependencies in self.meta.dependencies}

        hub_dependencies = {step_name: step_dependencies for step_name, step_dependencies in self.meta.hub_dependencies}
        
        for step_name, step_dependencies in raw_dependencies.items():
            if not step_dependencies and not hub_dependencies[step_name]:
                raw_dependencies[step_name] = set(('_source',))

        pprint("Raw dependencies")
        pprint(raw_dependencies)
        pprint("Hub dependencies")
        pprint(hub_dependencies)
        pprint(graph_steps)
        for iter_idx, step_info in enumerate(self.__iter__()):
            if step_info.name == "_source":
                continue

            pprint(f"step: {step_info.name}, dependencies: {raw_dependencies[step_info.name]}")
            console.rule("Iteration #{}".format(iter_idx))
            pprint(graph_steps)
            console.rule()
            if step_info.name in raw_dependencies:
                for dependent_step_name in raw_dependencies[step_info.name]:
                    pop_indices = []
                    if dependent_step_name and dependent_step_name != '_source':
                    #     dependent_step_name = '_source'
                    # else:
                        # Now we need to find
                        dependent_step_name = None

                        for step_idx, step in enumerate(graph_steps):
                            if step.step_name == step_info.name:
                                continue
                            
                            if step.is_related(study_graph[(step_info.name, step_info.workspace_name)]):
                                # pprint((study_graph[(step_info.name, step_info.workspace_name)], " IS related to ", step))
                                pprint(f"{step} is related to {study_graph[(step_info.name, step_info.workspace_name)]}")
                                dependent_step_name = (step.step_name, step.workspace_name)

                                # pprint(f"marking {step.step_name}:{step.workspace_name} for popping off the graph_steps list")                                
                                # pop_indices.append(step_idx)
                                # graph_steps.pop(step_idx)
                            # else:
                            #     pprint((study_graph[(step_info.name, step_info.workspace_name)], " IS NOT related to ", step))

                    if dependent_step_name:
                        console.rule("Adding Edge")
                        pprint(f"  {dependent_step_name} -> {(step_info.name, step_info.workspace_name)}")
                        study_graph.add_edge(dependent_step_name,
                                             (step_info.name, step_info.workspace_name))
                        
                    # pprint(f"len graph_steps: {len(graph_steps)}, pop_indices: {pop_indices}")
                    # for pop_idx in sorted(pop_indices, reverse=True):
                        
                    #     pprint(f"popping {graph_steps[pop_idx].step_name}:{graph_steps[pop_idx].workspace_name} off the graph_steps list")
                    #     graph_steps.pop(pop_idx)

            if step_info.name in hub_dependencies:
                for dependent_step_name in hub_dependencies[step_info.name]:
                    pprint("Processing hub dependency step {}".format(dependent_step_name))
                    # 
                    if not dependent_step_name:
                        continue

                    pop_indices = []
                    for step_idx, step in enumerate(graph_steps):
                        pprint(f"Procesing graph_step {step.step_name}:{step.workspace_name} in hub_dependencies check")
                        if step.step_name == step_info.name:
                            pprint("Found step {} in graph when looking for children of {}".format(step.step_name, step_info.name))
                            continue
                        elif step.step_name == dependent_step_name:
                            found_step_name = (step.step_name, step.workspace_name)
                            pprint(f"Adding Edge: {found_step_name} -> {(step_info.name, step_info.workspace_name)}")
                            study_graph.add_edge(found_step_name,
                                                 (step_info.name, step_info.workspace_name))
                            # pop_indices.append(step_idx)
                            # pprint(f"marking {step.step_name}:{step.workspace_name} for popping off the graph_steps list")
                    # pprint(f"len graph_steps: {len(graph_steps)}, pop_indices: {pop_indices}")
                    # for pop_idx in sorted(pop_indices, reverse=True):
                        
                    #     pprint(f"popping {graph_steps[pop_idx].step_name}:{graph_steps[pop_idx].workspace_name} off the graph_steps list")
                    #     graph_steps.pop(pop_idx)

        return study_graph



@define(kw_only=True)
class MaestroStep(object):
    """Container for maestro step info and comparison facilities"""
    step_name: str
    workspace_name: str
    workspace_path: str
    status: str                 # NOTE: make this a MaestroStatusEntry, or just store the state?
    param_combo_str: str
    param_list: Factory(list)

    @classmethod
    def create_step(cls, step_name=None, workspace_name=None, workspace_path=None, status=None, param_combo_str=None, param_dict=None):
        param_list = sorted([(param, value) for param, value in param_dict.items()],
                            key=lambda item: item[0])

        return cls(step_name=step_name,
                   workspace_name=workspace_name,
                   workspace_path=workspace_path,
                   status=status,
                   param_combo_str=param_combo_str,
                   param_list=param_list)

    def is_related(self, test_step):
        """Helper for determining if steps are related based on comparing param_list"""

        # for param_combo, test_param_combo in zip(self.param_list,
        #                                          test_step.param_list):
        #     pprint(f"{param_combo} == {test_param_combo}: {param_combo == test_param_combo}")
        return all([param_combo == test_param_combo
                    for param_combo, test_param_combo in zip(self.param_list,
                                                             test_step.param_list)])


@define(kw_only=True)
class MaestroTime(object):
    """Utility for working with times/durations in Maestro's status output"""
    days: int
    hours: int
    minutes: int
    seconds: int

@define(kw_only=True)
class MaestroDateTime(object):
    time: MaestroTime
    year: int
    day: int
    month: int
    
@define(kw_only=True)
class MaestroStatusEntry(object):
    step_name: str
    job_id: int
    workspace: str
    state: str
    run_time: MaestroTime
    elapsed_time: MaestroTime
    start_time: MaestroDateTime
    submit_time: MaestroDateTime
    end_time: MaestroDateTime
    num_restarts: int
    params: Factory(dict)


@define(kw_only=True)
class MaestroStatus(object):
    columns: Factory(list)
    rows: Factory(list)

    @classmethod
    def from_status_csv(cls, status_csv_dict):
        columns = [cls.make_safe(colname) for colname in status_csv_dict.keys()]

        rows = [dict(zip(columns, row_vals))
                for row_vals in zip(*status_csv_dict.values())]
        return cls(columns=columns,
                   rows=rows)

    @staticmethod
    def make_safe(name):
        return name.replace(' ', '_').lower()

    def row_by_workspace(self, workspace):
        for row in self.rows:
            if row['workspace'] == workspace:
                return row


class Meta(object):
    """Parses the Maestro meta files."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.metadata_path = os.path.join(self.path, 'metadata.yaml')
        self.parameters_path = os.path.join(self.path, 'parameters.yaml')
        self.environment_path = os.path.join(self.path, 'environment.yaml')
        self.study_path = os.path.join(self.path, 'study')

        self.metadata = load_yaml(self.metadata_path)

        #self.workspaces = self.metadata['workspaces']

        self.source = self.metadata['workspaces']['_source']

        self.parameter_map, self.parameters = self._get_parameters()

        # Setup graph for iteration
        self.study_graph = AcyclicAdjGraph()
        #  NOTE: REPLACE THIS WITH A STEP OBJECT INSTEAD OF NONE
        self.study_graph['_source'] = None

    @property
    def workspaces(self):
        """Generator yielding workspace, workspace_path"""
        for workspace, workspace_path in self._get_workspaces():
            yield workspace, workspace_path

    def _get_workspaces(self):
        """
        A list of workspaces in the study (excludes the _source dir)

        :yields: Workspaces in the study.
        """
        for key, value in self.metadata['workspaces'].items():
            if key != '_source':
                yield key, value

    @property
    def dependencies(self):
        for step_name, step_dependencies in self.metadata['dependencies'].items():
            if step_name == '_source':
                continue
            
            yield step_name, step_dependencies

    @property
    def hub_dependencies(self):
        for step_name, step_dependencies in self.metadata['hub_dependencies'].items():
            if step_name == '_source':
                continue
            
            yield step_name, step_dependencies

    def build_graph(self):
        """Uses dependencies/hub_dependencies to build the dag for dfs/bfs iteration"""
        # First loop through all the steps, adding them as nodes
        #   Need them to be data classes for comparison? -> param combo equality
        for step_name, stripped_step_combo, step_workspace in self._get_steps():
            self.study_graph[step_workspace] = {'step': step_name, 'param_combo': stripped_step_combo}
        
        # Start adding the edges using dependencies, starting by adding dependency free edges to _source
        visited_dependencies = {}
        raw_dependencies = {step_name: step_dependencies for step_name, step_dependencies in self.dependencies}
        
        for step_name, step_dependencies in raw_dependencies:
            if not step_dependencies:
                raw_dependencies[step_name] = set(('_source'))

        # Start visiting the dependencies and adding edges
        # for step_name, stripped_step_combo, step_workspace in self._get_steps():
            

    def find_step(self, search_step_name, search_step_combo):
        for step_name, stripped_step_combo, step_workspace in self._get_steps():
            if step_name == search_step_name and stripped_step_combo == search_step_combo:
                return step_name, stripped_step_combo, step_workspace

    def _get_steps(self):
        """
        Generator over study workspaces, excluding _source

        Yields
        ------
        tuple of step_name, step_param_combo_string, and step_workspace
        """
        for step_name, step_dict in self.metadata['step_combinations'].items():
            if step_name == '_source':
                continue

            # Strip off the step name prefix before yielding
            step_prefix = step_name + '_'
            #stripped_step_combos = [combo.replace(step_prefix, '') for combo in step_combos]

            for step_key, step_workspace in step_dict.items():
                stripped_step_combo = step_key.replace(step_prefix, '')
                yield step_name, stripped_step_combo, step_workspace

    def _get_parameters(self):
        """
        Creates mapping between param:workspace list in metadata.yaml and the param combo
        strings in parameters.yaml to facilitate correlating parameter values and labels
        with each workspace.

        Returns
        -------
        param_map : dict
            dict mapping metadata.yaml param combos to parameters.yaml parameter
            combo strings.  Acts as lookup index into the parsed parameters dict

        parameters : dict
            contains parsed labels and values for each parameter in each
            parameter combo. Top level keys are parameter combo strings,
            'labels' and 'values' are keys in subdict in each combo string
        """
        parameters = load_yaml(self.parameters_path)

        # Clean up the keys: strip the $( ) off, and the '.label' on the labels
        parameters = self.clean_parameters(parameters)

        # Check for ordering mismatches between parameter and meta files
        steps, stripped_step_combos = [], []

        for s, sc, s_path in self._get_steps():
            steps.append(s)
            stripped_step_combos.append(sc)

        # Build complete step_combo->param_combo map
        # NOTE: find out if params are indeed a reduced set
        param_map = {}
        stripped_combos = []
        for step, stripped_step_combo, step_path in self._get_steps():
            stripped_combos.append(stripped_step_combo)

            for param_combo, param_details in parameters.items():
                param_match = None
                if param_combo == step:  # no params, just single workspace
                    param_match = step
                elif param_combo == stripped_step_combo:
                    param_match = param_combo
                else:
                    # Get label names to get split points to inject whitespace
                    formatted_labels = param_details['labels'].values()
                    if all([stripped_step_combo.find(label) >= 0 for label in formatted_labels]):
                        param_match = param_combo

                if (param_match and param_match not in param_map):
                    param_map[stripped_step_combo] = param_match

        # Do a little error checking here, logging any unprocessed items
        for stripped_combo in set(stripped_combos):
            if stripped_combo not in param_map:
                print("Step combo string '{}' does not match any "
                      "parameter combo strings.".format(stripped_step_combo))

        return param_map, parameters

    def clean_parameters(self, parameters):
        """
        # Clean up the keys: strip the $( ) off, and the '.label' on the labels
        Cleans up/parses the parameters.yaml dicts of parameter values and labels
        for each combo string

        Returns
        -------
        dict :
            Keys are parameter combo strings from parameters.yaml, values are
            dict of parsed/cleaned up 'labels' and 'values'
        """
        stripped_params = {}

        for param_combo, param_info in parameters.items():
            new_param_info = {}
            for key, vals in param_info.items():
                if key == 'labels':
                    new_labels = {}
                    for label_tag, rendered_label in vals.items():
                        # Maybe better to hit this with a regex?
                        stripped_tag = label_tag.replace('$(', '').replace(')', '').replace('.label', '')
                        new_labels[stripped_tag] = rendered_label

                    new_param_info['labels'] = new_labels

                elif key == 'params':
                    new_params = {}
                    for param_tag, rendered_param in vals.items():
                        stripped_tag = param_tag.replace('$(', '').replace(')', '')
                        new_params[stripped_tag] = rendered_param

                    new_param_info['params'] = new_params

            stripped_params[param_combo] = new_param_info

        return stripped_params


class MyStudy(MaestroStudyMixin, object):
    """
    Provides an interface for iterating over steps in a Maestro along with the
    parameters fed into them from Maestro.
    """
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path)

        # Correlated lists of Maestro parameter dicts
        self._runs = None
        self._params = None

        # Unpack the meta info for easier accessibility
        self.metafile = self.meta

        # Step processing registration
        self.step_funcs = {}

        # Load the status
        self.status_info = self.status


    # def __iter__(self):
    #     """Iterator over workspaces, yielding StepInfo namedtuples"""
    #     for workspace_name, workspace_path in self.metafile._get_workspaces():
    #         step, params = self._find_param(workspace_path)
    #         step_status = self.status_info.row_by_workspace(workspace_path)
    #         yield StepInfo(step, workspace_name, workspace_path, params, step_status)

    # def _find_param(self, workspace_path):
    #     """Searches for workspace_path to get step name and parameters"""
    #     for step, stripped_combo, step_workspace in self.metafile._get_steps():
    #         if workspace_path == step_workspace:
    #             if stripped_combo in self.metafile.parameter_map:
    #                 params = self.metafile.parameters[
    #                     self.metafile.parameter_map[stripped_combo]]
    #             else:
    #                 params = None  # More useful to have an empty iterable downstream?

    #             return step, params

    def register_step_func(self, step_name, step_func):
        """
        Prototype to register processing functions to operate on specific step names

        NOTE: WIP and untested/unfinished
        """
        steps = [step_info[0] for step_info in self.metafile._get_steps()]
        if step_name in steps:
            self.step_funcs[step_name] = step_func

            return True
        else:
            return False        # add logger/approproate messages in real version


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Script to test out mixin based maestro study indexer.")

    parser.add_argument('path', type=str, default='.', nargs='*',
                        help="Paths to Maestro study output directories.")

    return parser


def main():
    """Main driver for testing this script, iterating over a study workspace"""
    parser = setup_argparse()

    args = vars(parser.parse_args())

    paths = []
    for path in args['path']:
        paths.append(os.path.abspath(path))

    def test_func(step_name, params):
        print('Processing {}'.format(step_name))
        print('Got params: {}'.format(params))

    studies = []
    for path in paths:
        study = MyStudy(path)
        print(os.path.split(path))
        console.rule(f"Study run/param list: {os.path.split(path)[-1]}")
        for items in study:     # items are StepInfo instances
            console.rule()
            pprint("Step: {}".format(items.name))
            pprint(("Workspace: ", items.workspace_name))
            pprint("Workspace path: {}".format(items.workspace_path))
            pprint(("Parameters: ", items.parameters))
            pprint(("Status: ", items.status))
            pprint("")

        # study_graph = study.study_graph
        # node_id = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        # node_map = {}           # step name/workspace tuple to node_id
        # for step_tuple in study_graph:
        #     node_map[step_tuple] = node_id.pop(0)

        # step_color_map = {'hello_world':'orange',
        #                   'bye_world': 'purple',
        #                   '_source': 'deeppink'}
        # console.rule("Node map")
        # pprint(node_map)
        # console.rule("DFS iteration")
        # for node, parent in DepthFirstSearch.search(study_graph, '_source'):
        #     pprint(f"Node: {node}, Parent: {parent}")

        # pprint(list(study_graph.edges()))
        # eolchar = "\\n"
        # with open(f'{study.name}.dot', 'w') as dotfile:
        #     dotfile.write(f'digraph "{study.name}" {{\n')
        #     for step_tuple in study_graph:
        #         label = []
        #         label.append(study_graph[step_tuple].step_name)
        #         pprint(f"Writing style for node {step_tuple}")
        #         for param_combo in study_graph[step_tuple].param_list:
        #             label.append(f'{param_combo[0]}: {param_combo[1]}')

        #         if step_tuple == '_source':
        #             c_idx = step_tuple
        #         else:
        #             c_idx = step_tuple[0]
        #         step_str = eolchar.join(label)
        #         dotfile.write(make_dot_node_str(node_map[step_tuple],
        #                                         study_graph[step_tuple].step_name,
        #                                         step_str) + '\n')
        #         # dotfile.write(f'{node_map[step_tuple]} [label="{eolchar.join(label)}" fontcolor="white" style="filled" color="{step_color_map[c_idx]}"]\n')
        #     for node, parent in BreadthFirstSearch.search(study_graph, '_source'):
        #         if not parent:
        #             print(f"{node_map[node]}")
        #             dotfile.write(node_map[node] + '\n')
        #         else:
        #             print(f"{node_map[parent]} -> {node_map[node]}")
        #             dotfile.write(f"{node_map[parent]} -> {node_map[node]}\n")

        #     dotfile.write("}")

if __name__ == '__main__':
    sys.exit(main())
