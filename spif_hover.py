"""
SPIF HOVER TOOLTIPS -- shared PyVista mouse-hover picking utility
====================================================================
Adds a live tooltip that follows the MOUSE MOVING over a render window (not
a click) and shows the per-face value + pass/fail status of whichever mesh
is under the cursor. Used by spif_visualize.py, spif_showcase_all.py, and
showcase_5axis.py so the VTK picking/event-wiring logic is written once
instead of duplicated per window.

Mechanism: a vtkCellPicker fired on every VTK 'MouseMoveEvent', matched
against the vtkActor objects RETURNED by plotter.add_mesh() (by Python
object identity). Matching on the actor rather than the input pv.PolyData
is deliberate: PyVista's mapper setup for a scalar-colored mesh (scalars=
+ cmap/clim) does not always leave picker.GetDataSet() pointing at the
exact same Python object you passed in, so matching on the dataset silently
failed for colored meshes while still working for plain-color ones (which
don't go through that path) -- the actor is always the same object VTK
renders, with no such ambiguity. Each subplot (single-row layouts only,
i.e. plotter.subplot(0, panel_idx), which is the only layout used across
this project's visualizers) gets its own persistent text actor (name=
'hover_tooltip') that updates in place -- no new actors are created per
mouse move.

Requires an interactive render window (a real plotter.show() call on a
machine with a display/GPU) -- silently does nothing useful in off-screen
or headless rendering, and has no effect on static screenshots.
"""

import vtk


def enable_hover_tooltip(plotter, panel_layers, position='lower_left', font_size=10):
    """
    plotter      : the pv.Plotter, after all add_mesh() calls for every
                   panel are done but BEFORE plotter.show().
    panel_layers : dict {panel_idx (int): [layer, layer, ...]}
                   panel_idx matches plotter.subplot(0, panel_idx) -- every
                   window in this project uses a single-row layout.
                   Each layer is a dict:
                     'actor'    : the vtkActor RETURNED by that panel's
                                  plotter.add_mesh(...) call (capture it:
                                  `actor = pl.add_mesh(...)`). Required.
                     'label_fn' : callable(local_cell_id) -> str | None.
                                  local_cell_id indexes into that actor's
                                  mesh's own cells (0-based, in the order its
                                  faces array was built) -- not the global
                                  face index. Return None/'' to show nothing.
    position     : add_text() position keyword for the tooltip box. Default
                   'lower_left' to stay clear of the upper-left stats text
                   and the scalar bar these windows put on the right.

    Multiple panels each keep their OWN last-shown tooltip text (PyVista
    manages actors per-renderer), so moving from panel A to panel B and back
    can leave A's tooltip showing whatever it last was rather than clearing
    when you're not hovering it -- a deliberate simplicity trade-off, not a
    bug: it avoids tracking focus/leave events for a minor cosmetic gain.
    """
    try:
        interactor = plotter.iren.interactor
    except AttributeError:
        return   # no interactor available (e.g. off-screen) -- no-op

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)

    actor_lookup = {}         # id(vtkActor) -> (panel_idx, label_fn)
    renderer_to_panel = {}    # id(vtkRenderer) -> panel_idx
    for panel_idx, layers in panel_layers.items():
        try:
            renderer_to_panel[id(plotter.renderers[panel_idx])] = panel_idx
        except (IndexError, TypeError):
            continue
        for layer in layers:
            actor = layer.get('actor')
            if actor is None:
                continue   # layer wasn't given a valid actor -- skip silently
            actor_lookup[id(actor)] = (panel_idx, layer['label_fn'])

    last_text = {}   # panel_idx -> last string shown (avoid redundant re-adds)

    def _set_text(panel_idx, text):
        if last_text.get(panel_idx) == text:
            return
        plotter.subplot(0, panel_idx)
        plotter.add_text(text, position=position, font_size=font_size,
                         color='black', name='hover_tooltip', shadow=True)
        last_text[panel_idx] = text

    def _on_move(caller, event):
        x, y = interactor.GetEventPosition()
        renderer = interactor.FindPokedRenderer(x, y)
        if renderer is None:
            return
        panel_idx = renderer_to_panel.get(id(renderer))

        picker.Pick(x, y, 0, renderer)
        picked_actor = picker.GetActor()
        cell_id = picker.GetCellId()
        match = actor_lookup.get(id(picked_actor)) if picked_actor is not None else None

        if match is None or cell_id < 0:
            if panel_idx is not None:
                _set_text(panel_idx, '')
            return

        owning_panel, label_fn = match
        text = label_fn(cell_id) or ''
        _set_text(owning_panel, text)

    interactor.AddObserver('MouseMoveEvent', _on_move)


def value_label(title, value, fmt='{:.1f}', unit='', pass_fail=None,
                pass_word='PASS', fail_word='FAIL', extra=''):
    """
    Small formatting helper for a consistent tooltip layout:

        {title}
        {value}{unit}
        {PASS/FAIL}      (only if pass_fail is not None)
        {extra}          (only if given)

    pass_fail : True/False -> appends a pass/fail line; None -> omitted
                (used for values with no pass/fail concept, e.g. force).
    """
    lines = [title, f"{fmt.format(value)}{unit}"]
    if pass_fail is not None:
        lines.append(pass_word if pass_fail else fail_word)
    if extra:
        lines.append(extra)
    return "\n".join(lines)
