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

        # Make optional arguments fields:
        from chimerax.ui.widgets import EntriesRow

        ds = EntriesRow(parent, 'Downsample', ('default (64)', '32', '64', '128', '256'))
        self._downsample_frame = ds.frame
        self._downsample = ds.values[0]

        prj = EntriesRow(parent, 'Projections', tuple(['default (30)'] + [str(i*10) for i in range(1, 10)]))
        self._projections_frame = prj.frame
        self._projections = prj.values[0]

        # Align button:
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

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
        self._object_menu = om = ModelMenuButton(self.session, class_filter=Volume)
        vlist = self.session.models.list(type=Volume)
        if vlist:
            om.value = vlist[0]
        mlayout.addWidget(om)

        iml = QLabel('to reference map', mf)
        mlayout.addWidget(iml)

        self._map_menu = mm = ModelMenuButton(self.session, class_filter=Volume)
        mlayout.addWidget(mm)
        if vlist:
            mm.value = vlist[-1]
        mlayout.addStretch(1)  # Extra space at end

        return mf

    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, [('align', self._emalign)], spacing=10, button_list=True)

        return f

    def _emalign(self):
        query_map = self._query_map()
        ref_map = self._map_menu.value

        if query_map is None:
            self.status('Choose model or map to align.')
            return
        if ref_map is None:
            self.status('Choose map to align to.')
            return
        if query_map == ref_map:
            self.status('Map to align must be different from map being aligned to.')
            return
        if self._downsample.value == 'default (64)':
            ds = 64
        else:
            ds = int(self._downsample.value)
        if self._projections.value == 'default (30)':
            prj = 30
        else:
            prj = int(self._projections.value)

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

        self._run_emalign(ref_map, query_map, ds, prj)

    def _run_emalign(self, ref_map, query_map, ds, prj):
        from .emalign_cmd import emalign
        emalign(self.session, ref_map, query_map, ds, prj)

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, EMalignDialog, 'EMalign', create=create)

    def status(self, message, log=False):
        self._status_label.setText(message)
        if log:
            self.session.logger.info(message)

    # @property
    # def _map(self):
    #     return self._map_menu.value

    # The query map chosen to align to the reference map:
    def _query_map(self):
        m = self._object_menu.value
        from chimerax.map import Volume
        return m if isinstance(m, Volume) else None


# ----------------------------------------------------------------------------------------------------------------------
def emalign_dialog(session, create=False):
    return EMalignDialog.get_singleton(session, create=create)


# ----------------------------------------------------------------------------------------------------------------------
def show_emalign_dialog(session):
    return emalign_dialog(session, create=True)
