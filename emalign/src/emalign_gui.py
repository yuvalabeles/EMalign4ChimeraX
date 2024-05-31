from chimerax.core.tools import ToolInstance


class EMalignDialog(ToolInstance):
    # help = 'help:user/tools/emalign.html'  # assure that contains help guide to EMalign

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
        # here will need to add try\except:
        vlist = self.session.models.list(type=Volume)
        om.value = vlist[0]
        # om.value_changed.connect(self._object_chosen)
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
        f, buttons = button_row(parent,
                                [('align', self._emalign),
                                 # ('Options', self._show_or_hide_options)
                                 ],
                                spacing=10,
                                button_list=True)

        return f

    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    def _create_options_gui(self, parent):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title=None)

        return p

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

    # query map chosen to align to ref map:
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
