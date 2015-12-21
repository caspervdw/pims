import os
import numpy as np
from pims import pipeline
from types import FunctionType
from skimage.viewer.widgets import Slider
from skimage.viewer.qt import Qt, QtWidgets, QtGui, QtCore, Signal, has_qt
from skimage.viewer.utils import (init_qtapp, figimage, start_qtapp,
                                  update_axes_image)


def available():
    try:
        if not has_qt:
            raise ImportError
    except ImportError:
        return False
    else:
        return True


dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


class DockWidgetCloseable(QtWidgets.QDockWidget):
    close_event_signal = Signal()
    def closeEvent(self, event):
        self.close_event_signal.emit()
        super(DockWidgetCloseable, self).closeEvent(event)


class BlitManager(object):
    """Object that manages blits on an axes"""
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('draw_event', self.on_draw_event)
        self.ax = ax
        self.background = None
        self.artists = []

    def add_artists(self, artists):
        self.artists.extend(artists)
        self.redraw()

    def remove_artists(self, artists):
        for artist in artists:
            self.artists.remove(artist)

    def on_draw_event(self, event=None):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_artists()

    def redraw(self):
        if self.background is not None:
            self.canvas.restore_region(self.background)
            self.draw_artists()
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def draw_artists(self):
        for artist in self.artists:
            self.ax.draw_artist(artist)


class Viewer(QtWidgets.QMainWindow):
    """Viewer for displaying image sequences from pims readers.

    Parameters
    ----------
    reader : pims reader, optional
        Reader that loads images to be displayed.
    """
    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    def __init__(self, reader=None, useblit=True):
        self.pipelines = []
        self.index = 0

        if reader is None:
            self.reader = np.zeros((1, 128, 128))
        else:
            self.reader = reader

        image = reader[0]
        # Start main loop
        init_qtapp()
        super(Viewer, self).__init__()

        #TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        self.fig, self.ax = figimage(image)
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)
        self.ax.autoscale(enable=False)

        self._tools = []
        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)

        self._image_plot = self.ax.images[0]
        self.original_image = image
        self.update_image()
        self.plugins = []

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        self.resize(cs_size.width(), cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)

        slider_kws = dict(value=0, low=0, high=len(self.reader) - 1)
        slider_kws['update_on'] = 'release'
        slider_kws['callback'] = lambda x, y: self.update_index(y)
        slider_kws['value_type'] = 'int'
        self.slider = Slider('index', **slider_kws)
        self.layout.addWidget(self.slider)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.pipeline_changed.connect(self.update_pipeline)
        plugin.attach(self)

        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = DockWidgetCloseable()
            dock.setWidget(plugin)
            dock.setWindowTitle(plugin.name)
            dock.close_event_signal.connect(plugin.cancel_pipeline)
            self.addDockWidget(dock_location, dock)

            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(plugin, dimension=dimension)

        return self

    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()

        dx = dy = 0
        if dimension == 'width':
            dx = widget_size.width()
        elif dimension == 'height':
            dy = widget_size.height()

        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_file(self, filename=None):
        """Open image file and display in viewer."""
        if filename is None:
            try:
                cur_dir = os.path.dirname(self.reader.filename)
            except AttributeError:
                cur_dir = ''
            filename = QtGui.QFileDialog.getOpenFileName(directory=cur_dir)
            if isinstance(filename, tuple):
                # Handle discrepancy between PyQt4 and PySide APIs.
                filename = filename[0]
        if filename is None or len(filename) == 0:
            return
        import pims
        reader = pims.open(filename)
        try:   # attempt to close current reader
            self.reader.close()
        except:
            pass
        self.reader = reader
        self.slider.slider.setRange(0, self.num_images - 1)
        self.slider.val = 0
        self.slider.editbox.setText('0')
        self.update_index(0)

    def update_index(self, index):
        """Select image on display using index into image collection."""
        index = min(max(int(round(index)), 0), len(self.reader) - 1)
        if index == self.index:
            return

        self.index = index
        self.slider.val = index
        image = self.reader[self.index]
        self.original_image = image
        self.update_image()

    def update_image(self):
        image = self.original_image.copy()
        for func in self.pipelines:
            image = func(image)
        self.image = image

    def update_pipeline(self, index, func):
        self.pipelines[index] = func
        self.update_image()

    def save_to_file(self, filename=None):
        raise NotImplementedError()

    def closeEvent(self, event):
        self.close()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self.move(0, 0)
        for p in self.plugins:
            p.show()
        super(Viewer, self).show()
        self.activateWindow()
        self.raise_()
        if main_window:
            start_qtapp()

        result = [None] * (len(self.pipelines) + 1)
        result[0] = self.reader
        for i, func in enumerate(self.pipelines):
            result[i + 1] = pipeline(func)(result[i])

        return result

    def redraw(self):
        if self.useblit:
            self._blit_manager.redraw()
        else:
            self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        update_axes_image(self._image_plot, image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = image.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._image_plot.set_clim(clim)

        if self.useblit:
            self._blit_manager.background = None

        self.redraw()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def _update_status_bar(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
            try:
                msg = "%4s @ [%4s, %4s]" % (self.image[y, x], x, y)
            except IndexError:
                msg = ""
        else:
            msg = ""
        self.status_message(msg)


class PipelinePlugin(QtWidgets.QDialog):
    """Base class for plugins that interact with an ImageViewer.

    A plugin connects an image filter (or another function) to an image viewer.
    Note that a Plugin is initialized *without* an image viewer and attached in
    a later step. See example below for details.

    Parameters
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement/manipulation.
    image_filter : function
        Function that gets called to update image in image viewer. This value
        can be `None` if, for example, you have a plugin that extracts
        information from an image and doesn't manipulate it. Alternatively,
        this function can be defined as a method in a Plugin subclass.
    height, width : int
        Size of plugin window in pixels. Note that Qt will automatically resize
        a window to fit components. So if you're adding rows of components, you
        can leave `height = 0` and just let Qt determine the final height.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        Matplotlib backends. If None, set to True when using Agg backend.
        This only has an effect if you draw on top of an image viewer.

    Attributes
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement.
    name : str
        Name of plugin. This is displayed as the window title.
    artist : list
        List of Matplotlib artists and canvastools. Any artists created by the
        plugin should be added to this list so that it gets cleaned up on
        close.

    Examples
    --------
    >>> from skimage.viewer import ImageViewer
    >>> from skimage.viewer.widgets import Slider
    >>> from skimage import data
    >>>
    >>> plugin = Plugin(image_filter=lambda img,
    ...                 threshold: img > threshold) # doctest: +SKIP
    >>> plugin += Slider('threshold', 0, 255)       # doctest: +SKIP
    >>>
    >>> image = data.coins()
    >>> viewer = ImageViewer(image)       # doctest: +SKIP
    >>> viewer += plugin                  # doctest: +SKIP
    >>> thresholded = viewer.show()[0][0] # doctest: +SKIP

    The plugin will automatically delegate parameters to `image_filter` based
    on its parameter type, i.e., `ptype` (widgets for required arguments must
    be added in the order they appear in the function). The image attached
    to the viewer is **automatically passed as the first argument** to the
    filter function.

    #TODO: Add flag so image is not passed to filter function by default.

    `ptype = 'kwarg'` is the default for most widgets so it's unnecessary here.

    """
    # Signals used when viewers are linked to the Plugin output.
    pipeline_changed = Signal(int, FunctionType)
    _started = Signal(int)

    def __init__(self, image_filter, name=None,
                 height=0, width=400, useblit=True,
                 dock='bottom'):
        init_qtapp()
        super(PipelinePlugin, self).__init__()

        self.dock = dock

        self.image_viewer = None
        self.image_filter = image_filter

        if name is None:
            self.name = image_filter.__name__
        else:
            self.name = name

        self.setWindowTitle(self.name)
        self.layout = QtWidgets.QGridLayout(self)
        self.resize(width, height)
        self.row = 0

        self.arguments = []
        self.keyword_arguments = {}

        self.useblit = useblit

    def attach(self, image_viewer):
        """Attach the plugin to an ImageViewer.

        Note that the ImageViewer will automatically call this method when the
        plugin is added to the ImageViewer. For example::

            viewer += Plugin(...)

        Also note that `attach` automatically calls the filter function so that
        the image matches the filtered value specified by attached widgets.
        """
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)

        self.pipeline_index = len(self.image_viewer.pipelines)
        self.image_viewer.pipelines += [None]

        self.rejected.connect(lambda: self.close())

        self.filter_image()

    def add_widget(self, widget):
        """Add widget to plugin.

        Alternatively, Plugin's `__add__` method is overloaded to add widgets::

            plugin += Widget(...)

        Widgets can adjust required or optional arguments of filter function or
        parameters for the plugin. This is specified by the Widget's `ptype`.
        """
        if widget.ptype == 'kwarg':
            name = widget.name.replace(' ', '_')
            self.keyword_arguments[name] = widget
            widget.callback = self.filter_image
        elif widget.ptype == 'arg':
            self.arguments.append(widget)
            widget.callback = self.filter_image
        elif widget.ptype == 'plugin':
            widget.callback = self.update_plugin
        widget.plugin = self
        self.layout.addWidget(widget, self.row, 0)
        self.row += 1

    def __add__(self, widget):
        self.add_widget(widget)
        return self

    def filter_image(self, *widget_arg):
        """Call `image_filter` with widget args and kwargs

        Note: `display_filtered_image` is automatically called.
        """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])
        func = lambda x: self.image_filter(x, *self.arguments, **kwargs)
        self.pipeline_changed.emit(self.pipeline_index, func)

    def _get_value(self, param):
        # If param is a widget, return its `val` attribute.
        return param if not hasattr(param, 'val') else param.val

    def update_plugin(self, name, value):
        """Update keyword parameters of the plugin itself.

        These parameters will typically be implemented as class properties so
        that they update the image or some other component.
        """
        setattr(self, name, value)

    def show(self, main_window=True):
        """Show plugin."""
        super(PipelinePlugin, self).show()
        self.activateWindow()
        self.raise_()

        # Emit signal with x-hint so new windows can be displayed w/o overlap.
        size = self.frameGeometry()
        x_hint = size.x() + size.width()
        self._started.emit(x_hint)

    def cancel_pipeline(self):
        """Close the plugin and clean up."""
        if self in self.image_viewer.plugins:
            self.image_viewer.plugins.remove(self)
        if self.pipeline_index is not None:
            del self.image_viewer.pipelines[self.pipeline_index]
        self.image_viewer.update_image()
        super(PipelinePlugin, self).close()
