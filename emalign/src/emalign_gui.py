from chimerax.core.tools import ToolInstance


class EMalignDialog(ToolInstance):
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
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

        # Status line:
        from Qt.QtWidgets import QLabel
        self._status_label = sl = QLabel(parent)
        layout.addWidget(sl)

        layout.addStretch(1)  # Extra space at end

        tw.manage(placement="side")

    def _create_emalign_map_menu(self, parent):
        from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel

        mf = QFrame(parent)
        mlayout = QHBoxLayout(mf)
        mlayout.setContentsMargins(0, 0, 0, 0)
        mlayout.setSpacing(10)

        fl = QLabel('Align query map', mf)
        mlayout.addWidget(fl)

        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelMenuButton
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
            rm.value = vlist[-1]

        rm.value_changed.connect(self._object_chosen)

        mlayout.addStretch(1)  # Extra space at end

        return mf

    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, [('align', self._emalign),
                                         ('Options', self._show_or_hide_options)], spacing=10, button_list=True)

        return f

    def _create_options_gui(self, parent):
        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title=None)
        f = p.content_area
        # self._v_size = None

        from chimerax.ui.widgets import EntriesRow, radio_buttons

        # EntriesRow(f, 'Downsample:', True, 'actual size', False, '64', False, '128', False, '256')
        EntriesRow(f, 'Downsample:')
        s_real = EntriesRow(f, True, 'actual size (<64)')
        s_64 = EntriesRow(f, False, '64')
        s_128 = EntriesRow(f, False, '128')
        s_256 = EntriesRow(f, False, '256')

        self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256 = s_real.values[0], s_64.values[0], s_128.values[0], s_256.values[0]

        radio_buttons(self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256)
        self._no_downsample_frame, self._downsample_64_frame, self._downsample_128_frame, self._downsample_256_frame = s_real.frame, s_64.frame, s_128.frame, s_256.frame

        from chimerax.map import Volume
        vlist = self.session.models.list(type=Volume)
        if vlist:
            self._update_options()
        else:
            self._no_downsample_frame.setEnabled(False)
            self._downsample_64_frame.setEnabled(False)
            self._downsample_128_frame.setEnabled(False)
            self._downsample_256_frame.setEnabled(False)
        # self._gray_out_downsample_options()

        per = EntriesRow(f, 'Projections:', False, '25 (fast)', True, '50 (default)', False, '125 (noisy data)')
        self._projections_25, self._projections_50, self._projections_125 = per.values
        radio_buttons(self._projections_25, self._projections_50, self._projections_125)
        self._projections_frame = per.frame

        if not vlist:
            self._projections_frame.setEnabled(False)
        return p

    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    def _emalign(self):
        query_map = self._query_map()
        ref_map = self._ref_map()

        if query_map is None:
            self.status('Choose model or map to align.')
            return
        if ref_map is None:
            self.status('Choose map to align to.')
            return
        if query_map == ref_map:
            self.status('Map to align must be different from map being aligned to.')
            return
        # if self._downsample.value == 'default (64)':
        #     ds = 64
        # else:
        #     ds = int(self._downsample.value)
        # if self._projections.value == 'default (30)':
        #     prj = 30
        # else:
        #     prj = int(self._projections.value)

        # if ds == '':
        #     self.status('Enter the downsampled size in pixels (default 64).')
        #     return
        # if ds.value <= 0 or ds.value > 256:
        #     self.status('Invalid downsample size. Must be between 1 and 256 (default 64).')
        #     return
        # if prj == '':
        #     self.status('Enter the number of projections (default 30).')
        #     return
        # if prj.value <= 0:
        #     self.status('Invalid projections number. Must be at least 1 (default 30).')
        #     return

        self._run_emalign(ref_map, query_map)

    def _run_emalign(self, ref_map, query_map):
        from .emalign_cmd import emalign
        emalign(self.session, ref_map, query_map)

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, EMalignDialog, 'EMalign', create=create)

    def status(self, message, log=False):
        self._status_label.setText(message)
        if log:
            self.session.logger.info(message)

    def _object_chosen(self):
        self._update_options()

    def _update_options(self):
        self._r_map = rm = self._ref_map()
        self._q_map = qm = self._query_map()
        if rm is None or qm is None:
            return
        self._assert_equal_size_volumes()
        self._v_size = max(rm.data.size)
        self._gray_out_downsample_options()

    def _assert_equal_size_volumes(self):
        if self._r_map.data.size != self._q_map.data.size:
            self.status('Map size mismatch.')
            return

    def _gray_out_downsample_options(self):
        # ref_map = self._map_menu.value
        # query_map = self._query_map()

        # ref_map_size = max(ref_map.data.size)
        v_size = self._v_size
        if v_size is None:
            self._no_downsample_frame.setEnabled(False)
            self._downsample_64_frame.setEnabled(False)
            self._downsample_128_frame.setEnabled(False)
            self._downsample_256_frame.setEnabled(False)
        elif v_size <= 64:
            self._no_downsample_frame.setEnabled(True)
            self._downsample_64_frame.setEnabled(False)
            self._downsample_128_frame.setEnabled(False)
            self._downsample_256_frame.setEnabled(False)

        elif 64 < v_size <= 128:
            self._no_downsample_frame.setEnabled(True)
            self._downsample_64_frame.setEnabled(True)
            self._downsample_128_frame.setEnabled(False)
            self._downsample_256_frame.setEnabled(False)
        elif 128 < v_size <= 256:
            self._no_downsample_frame.setEnabled(True)
            self._downsample_64_frame.setEnabled(True)
            self._downsample_128_frame.setEnabled(True)
            self._downsample_256_frame.setEnabled(False)
        else:
            self._no_downsample_frame.setEnabled(False)
            self._downsample_64_frame.setEnabled(True)
            self._downsample_128_frame.setEnabled(True)
            self._downsample_256_frame.setEnabled(True)

    def _ref_map(self):
        m = self._query_map_menu.value
        from chimerax.map import Volume
        return m if isinstance(m, Volume) else None

    # The query map chosen to align to the reference map:
    def _query_map(self):
        m = self._ref_map_menu.value
        from chimerax.map import Volume
        return m if isinstance(m, Volume) else None


# ----------------------------------------------------------------------------------------------------------------------
def emalign_dialog(session, create=False):
    return EMalignDialog.get_singleton(session, create=create)


# ----------------------------------------------------------------------------------------------------------------------
def show_emalign_dialog(session):
    return emalign_dialog(session, create=True)
