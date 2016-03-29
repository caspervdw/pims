from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from slicerator import pipeline
from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame
from pims.display import (export, play, scrollable_stack, to_rgb, normalize,
                          plot_to_frame, plots_to_frame)
from itertools import chain

import six
import glob
import os
from warnings import warn

from imageio import formats, get_reader
from imageio.core import Format

# has to be here for API stuff
from pims.image_sequence import ImageSequence, ImageSequenceND  # noqa
from .spe_stack import SpeStack


def not_available(requirement):
    def raiser(*args, **kwargs):
        raise ImportError(
            "This reader requires {0}.".format(requirement))
    return raiser


class PimsFormat(Format):
    """Wrapper for registering readers with ImageIO"""
    def __init__(self, *args, **kwargs):
        self._pims_reader = kwargs.pop('pims_reader')
        super(PimsFormat, self).__init__(*args, **kwargs)

    def _can_read(self, request):
        if request.mode[1] in (self.modes + '?'):
            if request.filename.lower().endswith(self.extensions):
                return True

    def _can_write(self, request):
        return False


    class Reader(Format.Reader):
        def _open(self, **kwargs):
            self._filename = self.request.get_local_filename()
            self._reader = self.format._pims_reader(self._filename, **kwargs)

        def _close(self):
            self._reader.close()

        def _get_length(self):
            return len(self._reader)

        def _get_data(self, index):
            frame = self._reader[index]
            return frame, frame.metadata

        def _get_meta_data(self, index):
            if index is None:
                return self._reader.metadata
            return self._reader[index].metadata

from .norpix_reader import NorpixSeq  # noqa
_fmt = PimsFormat(name='Norpix',
                  description='Read Norpix sequence (.seq) files',
                  extensions=' '.join(NorpixSeq.class_exts()),
                  modes='iv',
                  pims_reader=NorpixSeq)
formats.add_format(_fmt)

from .cine import Cine  # noqa
_fmt = PimsFormat(name='Cine',
                  description='Read cine files',
                  extensions=' '.join(Cine.class_exts()),
                  modes='iv',
                  pims_reader=Cine)
formats.add_format(_fmt)

try:
    import pims.pyav_reader
    if pims.pyav_reader.available():
        Video = pims.pyav_reader.PyAVVideoReader
    else:
        raise ImportError()
    _fmt = PimsFormat(name='PyAV',
                      description='Reads video files via PyAV.',
                      extensions=' '.join(Video.class_exts()),
                      modes='iI',
                      pims_reader=Video)
    formats.add_format(_fmt)
except (ImportError, IOError):
    Video = not_available("PyAV and/or PIL/Pillow")

import pims.tiff_stack
from pims.tiff_stack import (TiffStack_pil, TiffStack_libtiff,
                             TiffStack_tifffile)
# First, check if each individual class is available
# and drop in placeholders as needed.
if pims.tiff_stack.tifffile_available():
    _fmt = PimsFormat(name='Tifffile',
                      description='Reads .tiff files via tifffile.py.',
                      extensions=' '.join(TiffStack_tifffile.class_exts()),
                      modes='iv',
                      pims_reader=TiffStack_tifffile)
    formats.add_format(_fmt)
else:
    TiffStack_tifffile = not_available("tifffile")
if pims.tiff_stack.libtiff_available():
    _fmt = PimsFormat(name='Libtiff',
                      description='Reads .tiff files via libtiff',
                      extensions=' '.join(TiffStack_libtiff.class_exts()),
                      modes='iv',
                      pims_reader=TiffStack_libtiff)
    formats.add_format(_fmt)
else:
    TiffStack_libtiff = not_available("libtiff")
if pims.tiff_stack.PIL_available():
    _fmt = PimsFormat(name='PIL',
                      description='Reads .tiff files via PIL/Pillow',
                      extensions=' '.join(TiffStack_pil.class_exts()),
                      modes='iv',
                      pims_reader=TiffStack_pil)
    formats.add_format(_fmt)
else:
    TiffStack_pil = not_available("PIL or Pillow")
# Second, decide which class to assign to the
# TiffStack alias.
if pims.tiff_stack.tifffile_available():
    TiffStack = TiffStack_tifffile
elif pims.tiff_stack.libtiff_available():
    TiffStack = TiffStack_libtiff
elif pims.tiff_stack.PIL_available():
    TiffStack = TiffStack_pil
else:
    TiffStack = not_available("tifffile, libtiff, or PIL/Pillow")


try:
    import pims.bioformats
    if pims.bioformats.available():
        Bioformats = pims.bioformats.BioformatsReader
    else:
        raise ImportError()
    _fmt = PimsFormat(name='Bioformats',
                      description='Reads multidimensional images from filed '
                                  'supported by Bioformats.',
                      extensions=' '.join(Bioformats.class_exts()),
                      modes='iIvV',
                      pims_reader=Bioformats)
    formats.add_format(_fmt)
except (ImportError, IOError):
    Bioformats = not_available("JPype")


try:
    from pims_nd2 import ND2_Reader
    _fmt = PimsFormat(name='ND2_Reader',
                      description='Reads .nd2 images from NIS Elements images '
                                  'via the Nikon SDK.',
                      extensions=' '.join(ND2_Reader.class_exts()),
                      modes='iIvV',
                      pims_reader=ND2_Reader)
    formats.add_format(_fmt)
except ImportError:
    ND2_Reader = not_available("pims_nd2")


def open(sequence, **kwargs):
    """Read a filename, list of filenames, or directory of image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    sequence : string, list of strings, or glob
        The sequence you want to load. This can be a directory containing
        images, a glob ('/path/foo*.png') pattern of images,
        a video file, or a tiff stack
    kwargs :
        All keyword arguments will be passed to the reader.

    Examples
    --------
    >>> video = open('path/to/images/*.png')  # or *.tif, or *.jpg
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    files = glob.glob(sequence)
    if len(files) > 1:
        # todo: test if ImageSequence can read the image type,
        #       delegate to subclasses as needed
        return ImageSequence(sequence, **kwargs)

    # else: let ImageIO decide on the reader
    return ImageIOReader(sequence, **kwargs)


class ImageIOReader(FramesSequence):
    def __init__(self, filename, **kwargs):
        self.reader = get_reader(filename, **kwargs)
        self.filename = filename
        self._len = self.reader.get_length()

        first_frame = self.get_frame(0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

    def get_frame(self, i):
        frame = self.reader.get_data(i)
        return Frame(frame, frame_no=i, metadata=frame.meta)

    def get_metadata(self):
        return self.reader.get_meta_data(None)

    def __len__(self):
        return self._len

    def __iter__(self):
        iterable = self.reader.iter_data()
        for i in range(len(self)):
            frame = next(iterable)
            yield Frame(frame, frame_no=i, metadata=frame.meta)

    @property
    def frame_rate(self):
        return self.get_meta_data()['fps']

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype

    def close(self):
        self.reader.close()
