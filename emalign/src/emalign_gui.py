from chimerax.core import tools
from chimerax.core.tools import ToolInstance
from chimerax.map_fit import fitcmd
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import vertical_layout, button_row, ModelMenuButton, CollapsiblePanel, EntriesRow, radio_buttons
from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
from chimerax.map import Volume
from .emalign_cmd import emalign


class EMalignDialog(ToolInstance):
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        layout = vertical_layout(parent, margins=(5, 0, 0, 0))

        # Make menus to choose maps for alignment:
        mf = self._create_emalign_map_menu(parent)
        layout.addWidget(mf)

        # Align button:
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Make optional arguments fields:
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        # Make actions guide:
        guide = self._create_guide(parent)
        layout.addWidget(guide)

        # Status line:
        self._status_label = sl = QLabel(parent)
        layout.addWidget(sl)

        layout.addStretch(1)  # Extra space at end

        tw.manage(placement="side")

    def _create_emalign_map_menu(self, parent):
        mf = QFrame(parent)
        mlayout = QHBoxLayout(mf)
        mlayout.setContentsMargins(0, 0, 0, 0)
        mlayout.setSpacing(10)

        fl = QLabel('Align query map', mf)
        mlayout.addWidget(fl)

        self._query_map_menu = qm = ModelMenuButton(self.session, class_filter=Volume)
        vlist = self.session.models.list(type=Volume)
        if vlist:
            qm.value = vlist[0]
        qm.value_changed.connect(self._object_chosen)
        mlayout.addWidget(qm)

        iml = QLabel('to reference map', mf)
        mlayout.addWidget(iml)

        self._ref_map_menu = rm = ModelMenuButton(self.session, class_filter=Volume)
        mlayout.addWidget(rm)
        if vlist:
            rm.value = vlist[0]

        rm.value_changed.connect(self._object_chosen)

        mlayout.addStretch(1)  # Extra space at end

        return mf

    def _create_action_buttons(self, parent):
        f, buttons = button_row(parent, [('align', self._emalign),
                                         ('options', self._show_or_hide_options),
                                         ('help', self._show_or_hide_guide)], spacing=10, button_list=True)

        return f

    def _create_guide(self, parent):
        self._guide_panel = g = CollapsiblePanel(parent, title=None)
        f = g.content_area
        space = EntriesRow(f, ' ')
        downsample_guide = EntriesRow(f, 'Downsample - dimension to downsample input volumes to speed up computations. ')
        projection_guide = EntriesRow(f, 'Projections - number of projections to use for alignment. ')
        note = EntriesRow(f, '* the alignment may take a few minutes, don\'t click the screen while EMalign is running.')

        self.downsample_guide_frame = downsample_guide.frame
        self.projection_guide_frame = projection_guide.frame
        self.space_frame = space.frame
        self.note_frame = note.frame

        return g

    def _show_or_hide_guide(self):
        self._guide_panel.toggle_panel_display()

    def _create_options_gui(self, parent):
        self._options_panel = p = CollapsiblePanel(parent, title=None)
        f = p.content_area

        header_dns = EntriesRow(f, 'Downsample:')
        s_real = EntriesRow(f, True, 'None (use actual size)')
        s_64 = EntriesRow(f, False, '64')
        s_128 = EntriesRow(f, False, '128')
        s_256 = EntriesRow(f, False, '256')

        self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256 = s_real.values[0], s_64.values[0], s_128.values[0], s_256.values[0]
        radio_buttons(self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256)
        self._ds_frames = s_real.frame, s_64.frame, s_128.frame, s_256.frame
        self._no_downsample_frame, self._downsample_64_frame, self._downsample_128_frame, self._downsample_256_frame = self._ds_frames
        self._downsample_header_frame = header_dns.frame

        vlist = self.session.models.list(type=Volume)

        if not vlist:
            self._downsample_header_frame.setEnabled(False)
            self._no_downsample_frame.setEnabled(False)
            self._downsample_64_frame.setEnabled(False)
            self._downsample_128_frame.setEnabled(False)
            self._downsample_256_frame.setEnabled(False)

        header_proj = EntriesRow(f, 'Projections:')
        per = EntriesRow(f, False, '25 (fast)', True, '50 (default)', False, '125 (for noisier data)')
        self._projections_25, self._projections_50, self._projections_125 = per.values
        radio_buttons(self._projections_25, self._projections_50, self._projections_125)
        self._projections_frame = per.frame
        self.header_proj_frame = header_proj.frame

        use_fit_map = EntriesRow(f, True, 'Use Fit in Map to perform additional refinement (recommended)')
        self._use_fit_map = use_fit_map.values[0]
        self._use_fit_map_frame = use_fit_map.frame

        log = EntriesRow(f, False, 'Display detailed log')
        self._display_log = log.values[0]
        self._display_log_frame = log.frame

        params = EntriesRow(f, True, 'Display output parameters (rotation, translation, correlation)')
        self._display_parameters = params.values[0]
        self._display_parameters_frame = params.frame

        if not vlist:
            self.header_proj_frame.setEnabled(False)
            self._projections_frame.setEnabled(False)
            self._display_log_frame.setEnabled(False)
            self._display_parameters_frame.setEnabled(False)
            self._use_fit_map_frame.setEnabled(False)

        if vlist:
            self._update_options()
            
        return p

    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    def _emalign(self):
        query_map = self._query_map()
        ref_map = self._ref_map()

        if query_map is None:
            self.status('Choose map to align to the reference map.')
            return
        if ref_map is None:
            self.status('Choose map to align the query map to.')
            return
        if query_map == ref_map:
            self.status('The reference map must be different from the query map.')
            return

        downsample = self._get_downsample()
        projections = self._get_projections()
        show_log = self._display_log.value
        show_param = self._display_parameters.value

        self._run_emalign(ref_map, query_map, downsample, projections, show_log, show_param)

    def _run_emalign(self, ref_map, query_map, downsample, projections, show_log, show_param):
        emalign(self.session, ref_map, query_map, downsample=downsample, projections=projections, show_log=show_log, show_param=show_param, refine=False)
        if self._use_fit_map.value:
            # fitmap query_map inMap ref_map:
            fitcmd.fit_map_in_map(query_map, ref_map, metric='correlation', envelope=True, zeros=False, shift=True, rotate=True,
                                  move_whole_molecules=True, map_atoms=None, max_steps=2000, grid_step_min=0.01, grid_step_max=0.5)

    @classmethod
    def get_singleton(self, session, create=True):
        return tools.get_singleton(session, EMalignDialog, 'EMalign', create=create)

    def status(self, message, log=False):
        self._status_label.setText(message)
        if log:
            self.session.logger.info(message)

    def _object_chosen(self):
        self._update_options()
        self._status_label.setText(' ')

    def _update_options(self):
        self._r_map = rm = self._ref_map()
        self._q_map = qm = self._query_map()
        if rm is None or qm is None:
            return
        self._assert_equal_size_volumes()
        self._v_size = rm.data.size[0]
        self._gray_out_downsample_options()
        self._enable_other_options()

    def _assert_equal_size_volumes(self):
        if self._r_map.data.size != self._q_map.data.size:
            self.status('Map size mismatch.')
            return

    def _gray_out_downsample_options(self):
        v_size = self._v_size
        if v_size < 64:
            ds_values = [True, False, False, False]
        elif 64 <= v_size <= 128:
            ds_values = [True, True, False, False]
            self._no_downsample.value = False
            self._downsample_64.value = True
        elif 128 < v_size <= 256:
            ds_values = [True, True, True, True]
        else:
            ds_values = [False, True, True, True]
            self._no_downsample.value = False
            self._downsample_64.value = True

        for i in range(len(self._ds_frames)):
            self._ds_frames[i].setEnabled(ds_values[i])

    def _enable_other_options(self):
        options = [self._projections_frame, self._display_log_frame, self._display_parameters_frame, self._use_fit_map_frame, self._downsample_header_frame, self.header_proj_frame]
        for option in options:
            option.setEnabled(True)

    def _ref_map(self):
        m = self._ref_map_menu.value
        return m if isinstance(m, Volume) else None

    # The query map chosen to align to the reference map:
    def _query_map(self):
        m = self._query_map_menu.value
        return m if isinstance(m, Volume) else None

    def _get_downsample(self):
        downsample = 0
        if self._no_downsample.value:
            downsample = None
        if self._downsample_64.value:
            downsample = 64
        if self._downsample_128.value:
            downsample = 128
        if self._downsample_256.value:
            downsample = 256

        return downsample

    def _get_projections(self):
        projections = 0
        if self._projections_25.value:
            projections = 25
        if self._projections_50.value:
            projections = 50
        if self._projections_125.value:
            projections = 125

        return projections


# ----------------------------------------------------------------------------------------------------------------------
def emalign_dialog(session, create=False):
    return EMalignDialog.get_singleton(session, create=create)


# ----------------------------------------------------------------------------------------------------------------------
def show_emalign_dialog(session):
    return emalign_dialog(session, create=True)
