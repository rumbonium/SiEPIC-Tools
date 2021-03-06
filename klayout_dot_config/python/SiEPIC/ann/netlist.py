"""
netlist.py

Authors: 
    Sequoia Ploeg
    Hyrum Gunther

Dependencies:
- pya
    Python connection to KLayout, allows access to cells and other layout 
    objects. The custom netlist generator requires this.
- numpy
    Required for cascading s-matrices together.
- SiEPIC.extend, SiEPIC.core
    Required by the custom netlist generator.
- SiEPIC.ann.models
    The ObjectModelNetlist requires this module in order to build Component 
    models.
- jsons
    Similar to GSON in Java, serializes and deserializes custom models.
    Required to convert the ObjectModelNetlist to a file that can be saved and 
    read later.
    API: https://jsons.readthedocs.io/en/latest/index.html
- copy
    Some objects are deep copied during circuit matrix cascading.
- skrf
    Required for cascading s-parameter matrices.

This file contains everything related to netlist generation and modeling.
"""

import pya
# import SiEPIC.extend as se
# import SiEPIC.core as cor
from SiEPIC.ann.models.components import Component, create_component_by_name
import jsons
import copy
import numpy as np
import skrf as rf

class ObjectModelNetlist:
    """
    The Parser class reads a netlist generated by the SiEPIC toolbox and uses 
    various classes which inherit from 'models.components.Component' to create 
    an object based model of a photonic circuit. 
    
    Each derived class is connected to a component model in 'models' that 
    exposes a 'get_s_params' method with its appropriate arguments to the 
    derived model. These s_params are the s-matrices of the component, which 
    are then used to simulate the circuit's transmission behavior.

    Attributes
    ----------
    component_list : list
        A list of objects derived from 'models.components.Component' 
        representing the photonic circuit.
    net_count : int
        A counter keeping track of the total number of nets in the circuit 
        (0-indexed).
    json : string
        A JSON representation of the model. This is a property and cannot
        be set.

    Methods
    -------
    parse_file(filepath)
        Parses through the netlist to identify components and organize them 
        into objects. Objects are connected with their data models, allowing 
        them to retrieve any available parameters.
    _parse_line(line_elements)
        Reads the elements on a line of the netlist (already delimited before 
        passed to _parse_line) and creates the appropriate object. Appends the 
        newly created object to the Parser's component_list.
    """

    def __init__(self):
        """
        Initializes a Parser and creates a structure to hold a list of 
        components and count the number of nets in the circuit (0-indexed).
        """
        self.component_list = []
        self.net_count = 0


    def parse_file(self, filepath: str) -> list:
        """
        Parses a netlist (given a filename) and converts it to an object model 
        f the circuit.

        Parameters
        ----------
        filepath : str
            The name of the file to be parsed.

        Returns
        -------
        component_list : list
            A list of all components found in the netlist, with their 
            accompanying properties and values.
        """
        with open(filepath) as fid:
            text = fid.read()
            return self.parse_text(text)

    def parse_text(self, text: str) -> list:
        """
        Parses the string format of the netlist. Instead of requiring a file, 
        string representations of netlists can also be converted into an object
        model.

        Parameters
        ----------
        text : str
            The text of the netlist.
        
        Returns
        -------
        component_list : list
            A list of all components found in the netlist, with their 
            accompanying properties and values.
        """
        lines = text.splitlines()
        for line in lines:
                elements = line.split()
                if len(elements) > 0:
                    if (".ends" in elements[0]):
                        break
                    elif ("." in elements[0]) or ("*" in elements[0]):
                        continue
                    else:
                        self._parse_line(elements)
        return self.component_list

    def _parse_line(self, line_elements: list):
        """
        Parses a line from the netlist, already split into individual elements,
        and converts it into a new Component object.
        
        Parameters
        ----------
        line_elements : list
            A list of all the elements on a line (already split by some 
            delimiter).
        """

        # TODO: Consider having each component parse its own line, rather than
        # needing to add more case statements if new parameters show up.
        component = None
        nets = []
        for item in line_elements[1:]:
            if "N$" in item:
                net = str(item).replace("N$", '')
                nets.append(net)
                if int(net) > self.net_count:
                    self.net_count = int(net)
                continue
            elif component is None:
                component = create_component_by_name(item)
            elif "lay_x=" in item:
                component.lay_x = float(str(item).replace("lay_x=", ''))
            elif "lay_y=" in item:
                component.lay_y = float(str(item).replace("lay_y=", ''))
            elif "radius=" in item:
                component.radius = float(str(item).replace("radius=", ''))
            elif "wg_length=" in item:
                lenth = str(item).replace("wg_length=", '')
                component.length = strToSci(lenth)
            elif "wg_width=" in item:
                width = str(item).replace("wg_width=", '')
                # Width needs to be stored in microns (um)
                component.width = strToSci(width)*1e6
            elif "points=" in item:
                # The regex, in case you ever need it: /(\[[\d]+[\,][\d]+\])/g
                points = str(item).replace("points=", '')
                points = points.replace("\"[[", '')
                points = points.replace("]]\"", '')
                point_list = points.split('],[')
                for point in point_list:
                    out = point.split(',')
                    component.points.append((float(out[0]), float(out[1])))
        component.nets = nets
        self.component_list.append(component)

    def get_external_components(self):
        return [component for component in self.component_list if (any(int(x) < 0 for x in component.nets))]

    @property
    def json(self) -> str:
        return jsons.dump(self.component_list, verbose=True, strip_privates=True)
        # And, in case we ever want to build in a netlist export function,
        # here's the necessary code:
        # with open('data.json', 'w') as outfile:
        #     json.dump(output, outfile, indent=2)
        # with open('data.json') as jsonfile:
        #     data = json.load(jsonfile)
        # inputstr = jsons.load(data)


def spice_netlist_export(self) -> (str, str):
    """
    This function gathers information from the current top cell in Klayout into
    a netlist for a photonic circuit. This netlist is used in simulations.

    Code for this function is taken and adapted from a function in 
    'SiEPIC-Tools/klayout_dot_config/python/SiEPIC/extend.py' 
    which does the same thing, but to create a netlist for 
    Lumerical INTERCONNECT. This function has parts of that one removed 
    since they are not needed for this toolbox.
    """

    import SiEPIC
    from SiEPIC import _globals
    from time import strftime
    from SiEPIC.utils import eng_str

    from SiEPIC.utils import get_technology
    TECHNOLOGY = get_technology()
    if not TECHNOLOGY['technology_name']:
        v = pya.MessageBox.warning("Errors", "SiEPIC-Tools requires a technology to be chosen.  \n\nThe active technology is displayed on the bottom-left of the KLayout window, next to the T. \n\nChange the technology using KLayout File | Layout Properties, then choose Technology and find the correct one (e.g., EBeam, GSiP).", pya.MessageBox.Ok)
        return 'x', 'x', 0, [0]
    # get the netlist from the entire layout
    nets, components = self.identify_nets()

    if not components:
        v = pya.MessageBox.warning("Errors", "No components found.", pya.MessageBox.Ok)
        return 'no', 'components', 0, ['found']

    text_subckt = '* Spice output from KLayout SiEPIC-Tools v%s, %s.\n\n' % (
        SiEPIC.__version__, strftime("%Y-%m-%d %H:%M:%S"))
        
    circuit_name = self.name.replace('.', '')  # remove "."
    if '_' in circuit_name[0]:
        circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _

    ioports = -1
    for c in components:
        # optical nets: must be ordered electrical, optical IO, then optical
        nets_str = ''
        for p in c.pins:
            if p.type == _globals.PIN_TYPES.ELECTRICAL:
                nets_str += " " + c.component + '_' + str(c.idx) + '_' + p.pin_name
        for p in c.pins:
            if p.type == _globals.PIN_TYPES.OPTICALIO:
                nets_str += " N$" + str(ioports)
                ioports -= 1
        #pinIOtype = any([p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO])
        for p in c.pins:
            if p.type == _globals.PIN_TYPES.OPTICAL:
                if p.net.idx != None:
                    nets_str += " N$" + str(p.net.idx)
                #if p.net.idx != None:
                #    nets_str += " N$" + str(p.net.idx)
                else:
                    nets_str += " N$" + str(ioports)
                    ioports -= 1

        # Check to see if this component is an Optical IO type.
        pinIOtype = any([p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO])

        component1 = c.component
        params1 = c.params

        text_subckt += ' %s %s %s ' % (component1.replace(' ', '_') +
                                       "_" + str(c.idx), nets_str, component1.replace(' ', '_'))
        x, y = c.Dcenter.x, c.Dcenter.y
        text_subckt += '%s lay_x=%s lay_y=%s\n' % \
            (params1, eng_str(x * 1e-6), eng_str(y * 1e-6))

    om = ObjectModelNetlist()
    components = om.parse_text(text_subckt)
    output = jsons.dump(components, verbose=True, strip_privates=True)

    return text_subckt, output, om

pya.Cell.spice_netlist_export_ann = spice_netlist_export


def _match_ports(net_id: str, component_list: list) -> list:
    """
    Finds the components connected together by the specified net_id (string) in
    a list of components provided by the caller (even if the component is 
    connected to itself).

    Parameters
    ----------
    net_id : str
        The net id or name to which the components being searched for are 
        connected.
    component_list : list
        The complete list of components to be searched.

    Returns
    -------
    [comp1, netidx1, comp2, netidx2]
        A list (length 4) of integers with the following meanings:
        - comp1: Index of the first component in the list with a matching 
            net id.
        - netidx1: Index of the net in the ordered net list of 'comp1' 
            (corresponds to its column or row in the s-parameter matrix).
        - comp2: Index of the second component in the list with a matching 
            net id.
        - netidx1: Index of the net in the ordered net list of 'comp2' 
            (corresponds to its column or row in the s-parameter matrix).
    """
    filtered_comps = [component for component in component_list if net_id in component.nets]
    comp_idx = [component_list.index(component) for component in filtered_comps]
    net_idx = []
    for comp in filtered_comps:
        net_idx += [i for i, x in enumerate(comp.nets) if x == net_id]
    if len(comp_idx) == 1:
        comp_idx += comp_idx
    
    return [comp_idx[0], net_idx[0], comp_idx[1], net_idx[1]]


class ComponentSimulation:
    """
    This class is a simplified version of a Component in that it only contains
    an ordered list of nets, the frequency array, and the s-parameter matrix. 
    It can be initialized with or without a Component model, allowing its 
    attributes to be set after object creation.

    Attributes
    ----------
    nets : list(str)
        An ordered list of the nets connected to the Component
    f : np.array
        A numpy array of the frequency values in its simulation.
    s : np.array
        A numpy array of the s-parameter matrix for the given frequency range.
    """
    nets: list
    f: np.array
    s: np.array

    def __init__(self, component: Component=None):
        """
        Instantiates an object from a Component if provided; empty, if not.

        Parameters
        ----------
        component : Component, optional
            A component to initialize the data members of the object.
        """
        if component:
            self.nets = copy.deepcopy(component.nets)
            self.f, self.s = component.get_s_params()


def connect_circuit(netlist: ObjectModelNetlist) -> ComponentSimulation:
    """
    Connects the s-matrices of a photonic circuit given its ObjectModelNetlist
    and returns a single 'ComponentSimulation' object containing the frequency
    array, the assembled s-matrix, and a list of the external nets (strings of
    negative numbers).

    Returns
    -------
    ComponentSimulation
        After the circuit has been fully connected, the result is a single 
        ComponentSimulation with fields f (frequency), s (s-matrix), and nets 
        (external ports: negative numbers, as strings).
    """
    if netlist.net_count == 0:
        return

    component_list = [ComponentSimulation(component) for component in netlist.component_list]
    for n in range(0, netlist.net_count + 1):
        ca, ia, cb, ib = _match_ports(str(n), component_list)

        #if pin occurances are in the same Cell
        if ca == cb:
            component_list[ca].s = rf.innerconnect_s(component_list[ca].s, ia, ib)
            del component_list[ca].nets[ia]
            if ia < ib:
                del component_list[ca].nets[ib-1]
            else:
                del component_list[ca].nets[ib]

        #if pin occurances are in different Cells
        else:
            combination = ComponentSimulation()
            combination.f = component_list[0].f
            combination.s = rf.connect_s(component_list[ca].s, ia, component_list[cb].s, ib)
            del component_list[ca].nets[ia]
            del component_list[cb].nets[ib]
            combination.nets = component_list[ca].nets + component_list[cb].nets
            del component_list[ca]
            if ca < cb:
                del component_list[cb-1]
            else:
                del component_list[cb]
            component_list.append(combination)

    return component_list[0], netlist.get_external_components()


def strToSci(number) -> float:
    """
    Converts string representations of numbers written with abbreviated 
    prefixes into a float with the proper exponent (e.g. '3u' -> 3e-6).

    Parameters
    ----------
    number : str
        The number to be converted, represented as a string.
    
    Returns
    -------
    float
        The string converted to a float.
    """
    ex = number[-1]
    base = float(number[:-1])
    if(ex == 'm'):
        return base * 1e-3
    elif(ex == 'u'):
        return base * 1e-6
    elif(ex == 'n'):
        return base * 1e-9
    else:
        return float(number(base) + ex)

def get_sparameters(netlist: ObjectModelNetlist):
    """
    Gets the s-parameters matrix from a passed in ObjectModelNetlist by 
    connecting all components.

    Parameters
    ----------
    netlist: ObjectModelNetlist
        The netlist to be connected and have parameters extracted from.

    Returns
    -------
    np.array, np.array, list(str)
        A tuple in the following order: 
        ([s-matrix], [frequency array], [external port list])
        - s-matrix: The s-parameter matrix of the combined component.
        - frequency array: The corresponding frequency array, indexed the same
            as the s-matrix.
        - external port list: Strings of negative numbers representing the 
            ports of the combined component. They are indexed in the same order
            as the columns/rows of the s-matrix.
    """
    combined, edge_components = connect_circuit(netlist)
    f = combined.f
    s = combined.s
    externals = combined.nets
    return (s, f, externals, edge_components)
