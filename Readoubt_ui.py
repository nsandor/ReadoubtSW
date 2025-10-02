# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Readoubt.ui'
##
## Created by: Qt User Interface Compiler version 6.9.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMenu, QMenuBar, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QSplitter, QStackedWidget,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1074, 936)
        MainWindow.setMinimumSize(QSize(900, 700))
        self.actionConnect_SMU = QAction(MainWindow)
        self.actionConnect_SMU.setObjectName(u"actionConnect_SMU")
        self.actionConnect_Bias_SMU = QAction(MainWindow)
        self.actionConnect_Bias_SMU.setObjectName(u"actionConnect_Bias_SMU")
        self.actionConnect_SwitchBoard = QAction(MainWindow)
        self.actionConnect_SwitchBoard.setObjectName(u"actionConnect_SwitchBoard")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_0 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_0.setObjectName(u"verticalLayout_0")
        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.frame = QFrame(self.frame_2)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(0, 717))
        self.frame.setMaximumSize(QSize(358, 16777215))
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.scan_settings_box = QGroupBox(self.frame)
        self.scan_settings_box.setObjectName(u"scan_settings_box")
        self.scan_settings_box.setMinimumSize(QSize(0, 0))
        self.formLayout_2 = QFormLayout(self.scan_settings_box)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.Measurement_type_label = QLabel(self.scan_settings_box)
        self.Measurement_type_label.setObjectName(u"Measurement_type_label")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.Measurement_type_label)

        self.Measurement_type_combobox = QComboBox(self.scan_settings_box)
        self.Measurement_type_combobox.addItem("")
        self.Measurement_type_combobox.addItem("")
        self.Measurement_type_combobox.addItem("")
        self.Measurement_type_combobox.addItem("")
        self.Measurement_type_combobox.setObjectName(u"Measurement_type_combobox")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.Measurement_type_combobox)

        self.label_pixels = QLabel(self.scan_settings_box)
        self.label_pixels.setObjectName(u"label_pixels")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_pixels)

        self.edit_pixel_spec = QLineEdit(self.scan_settings_box)
        self.edit_pixel_spec.setObjectName(u"edit_pixel_spec")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.FieldRole, self.edit_pixel_spec)

        self.label_loops = QLabel(self.scan_settings_box)
        self.label_loops.setObjectName(u"label_loops")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_loops)

        self.spin_loops = QSpinBox(self.scan_settings_box)
        self.spin_loops.setObjectName(u"spin_loops")
        self.spin_loops.setMinimum(1)
        self.spin_loops.setMaximum(100000)
        self.spin_loops.setValue(1)

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.FieldRole, self.spin_loops)

        self.loop_delay_label = QLabel(self.scan_settings_box)
        self.loop_delay_label.setObjectName(u"loop_delay_label")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.loop_delay_label)

        self.loop_delay = QLineEdit(self.scan_settings_box)
        self.loop_delay.setObjectName(u"loop_delay")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.FieldRole, self.loop_delay)

        self.label_nplc = QLabel(self.scan_settings_box)
        self.label_nplc.setObjectName(u"label_nplc")

        self.formLayout_2.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_nplc)

        self.spin_nplc = QDoubleSpinBox(self.scan_settings_box)
        self.spin_nplc.setObjectName(u"spin_nplc")
        self.spin_nplc.setDecimals(2)
        self.spin_nplc.setMinimum(0.010000000000000)
        self.spin_nplc.setMaximum(25.000000000000000)
        self.spin_nplc.setValue(10.000000000000000)

        self.formLayout_2.setWidget(4, QFormLayout.ItemRole.FieldRole, self.spin_nplc)

        self.integration_time_label = QLabel(self.scan_settings_box)
        self.integration_time_label.setObjectName(u"integration_time_label")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.LabelRole, self.integration_time_label)

        self.integration_time = QLabel(self.scan_settings_box)
        self.integration_time.setObjectName(u"integration_time")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.FieldRole, self.integration_time)

        self.label_nsamp = QLabel(self.scan_settings_box)
        self.label_nsamp.setObjectName(u"label_nsamp")

        self.formLayout_2.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_nsamp)

        self.spin_nsamp = QSpinBox(self.scan_settings_box)
        self.spin_nsamp.setObjectName(u"spin_nsamp")
        self.spin_nsamp.setMinimum(1)
        self.spin_nsamp.setMaximum(100)
        self.spin_nsamp.setValue(1)

        self.formLayout_2.setWidget(6, QFormLayout.ItemRole.FieldRole, self.spin_nsamp)

        self.check_auto_current_range = QCheckBox(self.scan_settings_box)
        self.check_auto_current_range.setObjectName(u"check_auto_current_range")
        self.check_auto_current_range.setChecked(True)

        self.formLayout_2.setWidget(7, QFormLayout.ItemRole.LabelRole, self.check_auto_current_range)

        self.label_manual_rng = QLabel(self.scan_settings_box)
        self.label_manual_rng.setObjectName(u"label_manual_rng")

        self.formLayout_2.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_manual_rng)

        self.edit_current_range = QLineEdit(self.scan_settings_box)
        self.edit_current_range.setObjectName(u"edit_current_range")
        self.edit_current_range.setEnabled(False)

        self.formLayout_2.setWidget(8, QFormLayout.ItemRole.FieldRole, self.edit_current_range)


        self.verticalLayout.addWidget(self.scan_settings_box)

        self.save_box = QGroupBox(self.frame)
        self.save_box.setObjectName(u"save_box")
        self.save_layout = QFormLayout(self.save_box)
        self.save_layout.setObjectName(u"save_layout")
        self.label_exp = QLabel(self.save_box)
        self.label_exp.setObjectName(u"label_exp")

        self.save_layout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_exp)

        self.edit_exp_name = QLineEdit(self.save_box)
        self.edit_exp_name.setObjectName(u"edit_exp_name")

        self.save_layout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.edit_exp_name)

        self.label_out = QLabel(self.save_box)
        self.label_out.setObjectName(u"label_out")

        self.save_layout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_out)

        self.h_folder = QHBoxLayout()
        self.h_folder.setObjectName(u"h_folder")
        self.edit_output_folder = QLineEdit(self.save_box)
        self.edit_output_folder.setObjectName(u"edit_output_folder")
        self.edit_output_folder.setReadOnly(True)

        self.h_folder.addWidget(self.edit_output_folder)

        self.btn_browse_folder = QPushButton(self.save_box)
        self.btn_browse_folder.setObjectName(u"btn_browse_folder")

        self.h_folder.addWidget(self.btn_browse_folder)


        self.save_layout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.h_folder)

        self.label_autosave = QLabel(self.save_box)
        self.label_autosave.setObjectName(u"label_autosave")

        self.save_layout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_autosave)


        self.verticalLayout.addWidget(self.save_box)

        self.groupBox = QGroupBox(self.frame)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.plot_settings_stack = QTabWidget(self.groupBox)
        self.plot_settings_stack.setObjectName(u"plot_settings_stack")
        self.plot_settings_stackPage1 = QWidget()
        self.plot_settings_stackPage1.setObjectName(u"plot_settings_stackPage1")
        self.heatmap_settings_layout = QFormLayout(self.plot_settings_stackPage1)
        self.heatmap_settings_layout.setObjectName(u"heatmap_settings_layout")
        self.label_hm_title = QLabel(self.plot_settings_stackPage1)
        self.label_hm_title.setObjectName(u"label_hm_title")

        self.heatmap_settings_layout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_hm_title)

        self.edit_heatmap_title = QLineEdit(self.plot_settings_stackPage1)
        self.edit_heatmap_title.setObjectName(u"edit_heatmap_title")

        self.heatmap_settings_layout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.edit_heatmap_title)

        self.label_cmap = QLabel(self.plot_settings_stackPage1)
        self.label_cmap.setObjectName(u"label_cmap")

        self.heatmap_settings_layout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_cmap)

        self.combo_colormap = QComboBox(self.plot_settings_stackPage1)
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.addItem("")
        self.combo_colormap.setObjectName(u"combo_colormap")

        self.heatmap_settings_layout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.combo_colormap)

        self.label_log = QLabel(self.plot_settings_stackPage1)
        self.label_log.setObjectName(u"label_log")

        self.heatmap_settings_layout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_log)

        self.check_log_scale_heatmap = QCheckBox(self.plot_settings_stackPage1)
        self.check_log_scale_heatmap.setObjectName(u"check_log_scale_heatmap")
        self.check_log_scale_heatmap.setChecked(False)

        self.heatmap_settings_layout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.check_log_scale_heatmap)

        self.label_auto = QLabel(self.plot_settings_stackPage1)
        self.label_auto.setObjectName(u"label_auto")

        self.heatmap_settings_layout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_auto)

        self.check_auto_scale = QCheckBox(self.plot_settings_stackPage1)
        self.check_auto_scale.setObjectName(u"check_auto_scale")
        self.check_auto_scale.setChecked(False)

        self.heatmap_settings_layout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.check_auto_scale)

        self.label_manual_limits = QLabel(self.plot_settings_stackPage1)
        self.label_manual_limits.setObjectName(u"label_manual_limits")

        self.heatmap_settings_layout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_manual_limits)

        self.minmax_layout = QHBoxLayout()
        self.minmax_layout.setObjectName(u"minmax_layout")
        self.label_vmin = QLabel(self.plot_settings_stackPage1)
        self.label_vmin.setObjectName(u"label_vmin")

        self.minmax_layout.addWidget(self.label_vmin)

        self.edit_vmin = QLineEdit(self.plot_settings_stackPage1)
        self.edit_vmin.setObjectName(u"edit_vmin")
        self.edit_vmin.setEnabled(True)

        self.minmax_layout.addWidget(self.edit_vmin)

        self.label_vmax = QLabel(self.plot_settings_stackPage1)
        self.label_vmax.setObjectName(u"label_vmax")

        self.minmax_layout.addWidget(self.label_vmax)

        self.edit_vmax = QLineEdit(self.plot_settings_stackPage1)
        self.edit_vmax.setObjectName(u"edit_vmax")
        self.edit_vmax.setEnabled(True)

        self.minmax_layout.addWidget(self.edit_vmax)


        self.heatmap_settings_layout.setLayout(5, QFormLayout.ItemRole.FieldRole, self.minmax_layout)

        self.label_showvals = QLabel(self.plot_settings_stackPage1)
        self.label_showvals.setObjectName(u"label_showvals")

        self.heatmap_settings_layout.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_showvals)

        self.check_show_values = QCheckBox(self.plot_settings_stackPage1)
        self.check_show_values.setObjectName(u"check_show_values")
        self.check_show_values.setChecked(True)

        self.heatmap_settings_layout.setWidget(6, QFormLayout.ItemRole.FieldRole, self.check_show_values)

        self.label_export_hm = QLabel(self.plot_settings_stackPage1)
        self.label_export_hm.setObjectName(u"label_export_hm")

        self.heatmap_settings_layout.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_export_hm)

        self.btn_export_heatmap = QPushButton(self.plot_settings_stackPage1)
        self.btn_export_heatmap.setObjectName(u"btn_export_heatmap")

        self.heatmap_settings_layout.setWidget(7, QFormLayout.ItemRole.FieldRole, self.btn_export_heatmap)

        self.label_units = QLabel(self.plot_settings_stackPage1)
        self.label_units.setObjectName(u"label_units")

        self.heatmap_settings_layout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_units)

        self.combo_units = QComboBox(self.plot_settings_stackPage1)
        self.combo_units.addItem("")
        self.combo_units.addItem("")
        self.combo_units.addItem("")
        self.combo_units.addItem("")
        self.combo_units.setObjectName(u"combo_units")

        self.heatmap_settings_layout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.combo_units)

        self.plot_settings_stack.addTab(self.plot_settings_stackPage1, "")
        self.plot_settings_stackPage2 = QWidget()
        self.plot_settings_stackPage2.setObjectName(u"plot_settings_stackPage2")
        self.hist_settings_layout = QFormLayout(self.plot_settings_stackPage2)
        self.hist_settings_layout.setObjectName(u"hist_settings_layout")
        self.label_hist_title = QLabel(self.plot_settings_stackPage2)
        self.label_hist_title.setObjectName(u"label_hist_title")

        self.hist_settings_layout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_hist_title)

        self.edit_hist_title = QLineEdit(self.plot_settings_stackPage2)
        self.edit_hist_title.setObjectName(u"edit_hist_title")

        self.hist_settings_layout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.edit_hist_title)

        self.label_bins = QLabel(self.plot_settings_stackPage2)
        self.label_bins.setObjectName(u"label_bins")

        self.hist_settings_layout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_bins)

        self.spin_bins = QSpinBox(self.plot_settings_stackPage2)
        self.spin_bins.setObjectName(u"spin_bins")
        self.spin_bins.setMinimum(5)
        self.spin_bins.setMaximum(200)
        self.spin_bins.setValue(25)

        self.hist_settings_layout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.spin_bins)

        self.label_logy = QLabel(self.plot_settings_stackPage2)
        self.label_logy.setObjectName(u"label_logy")

        self.hist_settings_layout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_logy)

        self.check_log_scale_hist = QCheckBox(self.plot_settings_stackPage2)
        self.check_log_scale_hist.setObjectName(u"check_log_scale_hist")

        self.hist_settings_layout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.check_log_scale_hist)

        self.label_export_hist = QLabel(self.plot_settings_stackPage2)
        self.label_export_hist.setObjectName(u"label_export_hist")

        self.hist_settings_layout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_export_hist)

        self.btn_export_hist = QPushButton(self.plot_settings_stackPage2)
        self.btn_export_hist.setObjectName(u"btn_export_hist")

        self.hist_settings_layout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.btn_export_hist)

        self.plot_settings_stack.addTab(self.plot_settings_stackPage2, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.formLayout = QFormLayout(self.tab)
        self.formLayout.setObjectName(u"formLayout")
        self.label_ref = QLabel(self.tab)
        self.label_ref.setObjectName(u"label_ref")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_ref)

        self.ref_row = QHBoxLayout()
        self.ref_row.setObjectName(u"ref_row")
        self.btn_load_ref = QPushButton(self.tab)
        self.btn_load_ref.setObjectName(u"btn_load_ref")

        self.ref_row.addWidget(self.btn_load_ref)

        self.lbl_ref_info = QLabel(self.tab)
        self.lbl_ref_info.setObjectName(u"lbl_ref_info")

        self.ref_row.addWidget(self.lbl_ref_info)


        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.ref_row)

        self.label_op = QLabel(self.tab)
        self.label_op.setObjectName(u"label_op")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_op)

        self.combo_math = QComboBox(self.tab)
        self.combo_math.addItem("")
        self.combo_math.addItem("")
        self.combo_math.addItem("")
        self.combo_math.setObjectName(u"combo_math")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.combo_math)

        self.label_save_proc = QLabel(self.tab)
        self.label_save_proc.setObjectName(u"label_save_proc")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_save_proc)

        self.check_save_processed = QCheckBox(self.tab)
        self.check_save_processed.setObjectName(u"check_save_processed")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.check_save_processed)

        self.plot_settings_stack.addTab(self.tab, "")

        self.gridLayout.addWidget(self.plot_settings_stack, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox)


        self.horizontalLayout.addWidget(self.frame)

        self.centralSplitter = QSplitter(self.frame_2)
        self.centralSplitter.setObjectName(u"centralSplitter")
        self.centralSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.plotArea = QWidget(self.centralSplitter)
        self.plotArea.setObjectName(u"plotArea")
        self.plot_area_layout = QVBoxLayout(self.plotArea)
        self.plot_area_layout.setObjectName(u"plot_area_layout")
        self.plot_area_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_selector = QComboBox(self.plotArea)
        self.plot_selector.addItem("")
        self.plot_selector.addItem("")
        self.plot_selector.setObjectName(u"plot_selector")

        self.plot_area_layout.addWidget(self.plot_selector)

        self.plot_stack = QStackedWidget(self.plotArea)
        self.plot_stack.setObjectName(u"plot_stack")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_stack.sizePolicy().hasHeightForWidth())
        self.plot_stack.setSizePolicy(sizePolicy)
        self.plot_stack.setMinimumSize(QSize(500, 500))
        self.heatmap_page = QWidget()
        self.heatmap_page.setObjectName(u"heatmap_page")
        self.heatmap_page_layout = QVBoxLayout(self.heatmap_page)
        self.heatmap_page_layout.setObjectName(u"heatmap_page_layout")
        self.layout_canvas_heatmap = QVBoxLayout()
        self.layout_canvas_heatmap.setObjectName(u"layout_canvas_heatmap")

        self.heatmap_page_layout.addLayout(self.layout_canvas_heatmap)

        self.plot_stack.addWidget(self.heatmap_page)
        self.hist_page = QWidget()
        self.hist_page.setObjectName(u"hist_page")
        self.hist_page_layout = QVBoxLayout(self.hist_page)
        self.hist_page_layout.setObjectName(u"hist_page_layout")
        self.layout_canvas_hist = QVBoxLayout()
        self.layout_canvas_hist.setObjectName(u"layout_canvas_hist")

        self.hist_page_layout.addLayout(self.layout_canvas_hist)

        self.plot_stack.addWidget(self.hist_page)

        self.plot_area_layout.addWidget(self.plot_stack)

        self.centralSplitter.addWidget(self.plotArea)

        self.horizontalLayout.addWidget(self.centralSplitter)


        self.verticalLayout_0.addWidget(self.frame_2)

        self.ScanprogressBar = QProgressBar(self.centralwidget)
        self.ScanprogressBar.setObjectName(u"ScanprogressBar")
        self.ScanprogressBar.setValue(0)

        self.verticalLayout_0.addWidget(self.ScanprogressBar)

        self.ScanTimers = QLabel(self.centralwidget)
        self.ScanTimers.setObjectName(u"ScanTimers")
        self.ScanTimers.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_0.addWidget(self.ScanTimers)

        self.run_layout = QHBoxLayout()
        self.run_layout.setObjectName(u"run_layout")
        self.btn_run_abort = QPushButton(self.centralwidget)
        self.btn_run_abort.setObjectName(u"btn_run_abort")

        self.run_layout.addWidget(self.btn_run_abort)

        self.btn_pause_resume = QPushButton(self.centralwidget)
        self.btn_pause_resume.setObjectName(u"btn_pause_resume")
        self.btn_pause_resume.setEnabled(False)

        self.run_layout.addWidget(self.btn_pause_resume)


        self.verticalLayout_0.addLayout(self.run_layout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1074, 22))
        self.menuHardware = QMenu(self.menuBar)
        self.menuHardware.setObjectName(u"menuHardware")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuHardware.menuAction())
        self.menuHardware.addAction(self.actionConnect_SMU)
        self.menuHardware.addAction(self.actionConnect_Bias_SMU)
        self.menuHardware.addAction(self.actionConnect_SwitchBoard)

        self.retranslateUi(MainWindow)

        self.plot_settings_stack.setCurrentIndex(0)
        self.combo_units.setCurrentIndex(1)
        self.plot_stack.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Photodiode Array \u2013 Live Current Heat-map", None))
        self.actionConnect_SMU.setText(QCoreApplication.translate("MainWindow", u"Connect Read SMU", None))
        self.actionConnect_Bias_SMU.setText(QCoreApplication.translate("MainWindow", u"Connect Bias SMU", None))
        self.actionConnect_SwitchBoard.setText(QCoreApplication.translate("MainWindow", u"Connect SwitchBoard", None))
        self.scan_settings_box.setTitle(QCoreApplication.translate("MainWindow", u"Scan Settings", None))
        self.Measurement_type_label.setText(QCoreApplication.translate("MainWindow", u"Measurement Type", None))
        self.Measurement_type_combobox.setItemText(0, QCoreApplication.translate("MainWindow", u"Current - Time", None))
        self.Measurement_type_combobox.setItemText(1, QCoreApplication.translate("MainWindow", u"Current - Voltage", None))
        self.Measurement_type_combobox.setItemText(2, QCoreApplication.translate("MainWindow", u"Imaging", None))
        self.Measurement_type_combobox.setItemText(3, "")

        self.label_pixels.setText(QCoreApplication.translate("MainWindow", u"Pixels to scan:", None))
        self.edit_pixel_spec.setText(QCoreApplication.translate("MainWindow", u"1-100", None))
        self.edit_pixel_spec.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Ranges and indices, e.g., 50-70, 1, 10-12", None))
        self.label_loops.setText(QCoreApplication.translate("MainWindow", u"Loops:", None))
        self.loop_delay_label.setText(QCoreApplication.translate("MainWindow", u"Delay Between Loops (s)", None))
        self.loop_delay.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_nplc.setText(QCoreApplication.translate("MainWindow", u"NPLC:", None))
        self.integration_time_label.setText(QCoreApplication.translate("MainWindow", u"Integration Time:", None))
        self.integration_time.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_nsamp.setText(QCoreApplication.translate("MainWindow", u"Samples / pixel:", None))
        self.check_auto_current_range.setText(QCoreApplication.translate("MainWindow", u"Auto current range", None))
        self.label_manual_rng.setText(QCoreApplication.translate("MainWindow", u"Manual range (A):", None))
        self.edit_current_range.setText(QCoreApplication.translate("MainWindow", u"1e-7", None))
        self.save_box.setTitle(QCoreApplication.translate("MainWindow", u"Output & Saving", None))
        self.label_exp.setText(QCoreApplication.translate("MainWindow", u"Experiment Name:", None))
        self.edit_exp_name.setText(QCoreApplication.translate("MainWindow", u"MyExperiment", None))
        self.label_out.setText(QCoreApplication.translate("MainWindow", u"Output Folder:", None))
        self.btn_browse_folder.setText(QCoreApplication.translate("MainWindow", u"Browse\u2026", None))
        self.label_autosave.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Visualization Settings", None))
        self.label_hm_title.setText(QCoreApplication.translate("MainWindow", u"Title:", None))
        self.edit_heatmap_title.setText(QCoreApplication.translate("MainWindow", u"Photodiode Current", None))
        self.label_cmap.setText(QCoreApplication.translate("MainWindow", u"Colormap:", None))
        self.combo_colormap.setItemText(0, QCoreApplication.translate("MainWindow", u"inferno", None))
        self.combo_colormap.setItemText(1, QCoreApplication.translate("MainWindow", u"viridis", None))
        self.combo_colormap.setItemText(2, QCoreApplication.translate("MainWindow", u"plasma", None))
        self.combo_colormap.setItemText(3, QCoreApplication.translate("MainWindow", u"magma", None))
        self.combo_colormap.setItemText(4, QCoreApplication.translate("MainWindow", u"cividis", None))
        self.combo_colormap.setItemText(5, QCoreApplication.translate("MainWindow", u"gray_r", None))
        self.combo_colormap.setItemText(6, QCoreApplication.translate("MainWindow", u"jet", None))

        self.label_log.setText("")
        self.check_log_scale_heatmap.setText(QCoreApplication.translate("MainWindow", u"Logarithmic Color Scale", None))
        self.label_auto.setText("")
        self.check_auto_scale.setText(QCoreApplication.translate("MainWindow", u"Auto-scale Color Limit", None))
        self.label_manual_limits.setText(QCoreApplication.translate("MainWindow", u"Manual Limits:", None))
        self.label_vmin.setText(QCoreApplication.translate("MainWindow", u"Min:", None))
        self.edit_vmin.setText(QCoreApplication.translate("MainWindow", u"0.1", None))
        self.label_vmax.setText(QCoreApplication.translate("MainWindow", u"Max:", None))
        self.edit_vmax.setText(QCoreApplication.translate("MainWindow", u"10", None))
        self.label_showvals.setText("")
        self.check_show_values.setText(QCoreApplication.translate("MainWindow", u"Show Pixel Values", None))
        self.label_export_hm.setText("")
        self.btn_export_heatmap.setText(QCoreApplication.translate("MainWindow", u"Export Heatmap PNG\u2026", None))
        self.label_units.setText(QCoreApplication.translate("MainWindow", u"Units", None))
        self.combo_units.setItemText(0, QCoreApplication.translate("MainWindow", u"pA", None))
        self.combo_units.setItemText(1, QCoreApplication.translate("MainWindow", u"nA", None))
        self.combo_units.setItemText(2, QCoreApplication.translate("MainWindow", u"uA", None))
        self.combo_units.setItemText(3, QCoreApplication.translate("MainWindow", u"mA", None))

        self.plot_settings_stack.setTabText(self.plot_settings_stack.indexOf(self.plot_settings_stackPage1), QCoreApplication.translate("MainWindow", u"Heatmap", None))
        self.label_hist_title.setText(QCoreApplication.translate("MainWindow", u"Title:", None))
        self.edit_hist_title.setText(QCoreApplication.translate("MainWindow", u"Current Distribution", None))
        self.label_bins.setText(QCoreApplication.translate("MainWindow", u"Number of Bins:", None))
        self.label_logy.setText("")
        self.check_log_scale_hist.setText(QCoreApplication.translate("MainWindow", u"Logarithmic Y-Axis", None))
        self.label_export_hist.setText("")
        self.btn_export_hist.setText(QCoreApplication.translate("MainWindow", u"Export Histogram PNG\u2026", None))
        self.plot_settings_stack.setTabText(self.plot_settings_stack.indexOf(self.plot_settings_stackPage2), QCoreApplication.translate("MainWindow", u"Histogram", None))
        self.label_ref.setText(QCoreApplication.translate("MainWindow", u"Reference:", None))
        self.btn_load_ref.setText(QCoreApplication.translate("MainWindow", u"Load Ref CSV\u2026", None))
        self.lbl_ref_info.setText(QCoreApplication.translate("MainWindow", u"(none)", None))
        self.label_op.setText(QCoreApplication.translate("MainWindow", u"Operation:", None))
        self.combo_math.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.combo_math.setItemText(1, QCoreApplication.translate("MainWindow", u"Divide (live / ref)", None))
        self.combo_math.setItemText(2, QCoreApplication.translate("MainWindow", u"Subtract (live - ref)", None))

        self.label_save_proc.setText("")
        self.check_save_processed.setText(QCoreApplication.translate("MainWindow", u"Save processed CSVs instead of raw", None))
        self.plot_settings_stack.setTabText(self.plot_settings_stack.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Math", None))
        self.plot_selector.setItemText(0, QCoreApplication.translate("MainWindow", u"Heatmap", None))
        self.plot_selector.setItemText(1, QCoreApplication.translate("MainWindow", u"Histogram", None))

        self.ScanTimers.setText(QCoreApplication.translate("MainWindow", u"Loop #:                    Time Elapsed:                   Predicted remaining time:", None))
        self.btn_run_abort.setText(QCoreApplication.translate("MainWindow", u"Run Scan", None))
        self.btn_pause_resume.setText(QCoreApplication.translate("MainWindow", u"Pause", None))
        self.menuHardware.setTitle(QCoreApplication.translate("MainWindow", u"Hardware", None))
    # retranslateUi

