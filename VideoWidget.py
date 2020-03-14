
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QImage, QPalette, QPainter, QRegion
from PyQt5.QtMultimedia import QAbstractVideoSurface, QAbstractVideoBuffer, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtWidgets import QWidget, QSizePolicy


class VideoWidgetSurface(QAbstractVideoSurface):

    def __init__(self, widget, parent=None):
        super(VideoWidgetSurface, self).__init__(parent)
        self.parent = parent
        self.widget = widget
        self.output = None
        self.t_const = None
        self.i = 0
        self.imageFormat = QImage.Format_Invalid

    def supportedPixelFormats(self, handleType=QAbstractVideoBuffer.NoHandle):
        formats = [QVideoFrame.PixelFormat()]
        if (handleType == QAbstractVideoBuffer.NoHandle):
            for f in [QVideoFrame.Format_RGB32,
                      QVideoFrame.Format_ARGB32,
                      QVideoFrame.Format_ARGB32_Premultiplied,
                      QVideoFrame.Format_RGB565,
                      QVideoFrame.Format_RGB555
                      ]:
                formats.append(f)
        return formats

    def isFormatSupported(self, _format):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(_format.pixelFormat())
        size = _format.frameSize()
        _bool = False
        if (imageFormat != QImage.Format_Invalid and not
        size.isEmpty() and
                _format.handleType() == QAbstractVideoBuffer.NoHandle):
            _bool = True
        return _bool

    def start(self, _format):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(_format.pixelFormat())
        size = _format.frameSize()
        if (imageFormat != QImage.Format_Invalid and not size.isEmpty()):
            self.imageFormat = imageFormat
            self.imageSize = size
            self.sourceRect = _format.viewport()
            QAbstractVideoSurface.start(self, _format)
            self.widget.updateGeometry()
            self.updateVideoRect()
            return True
        else:
            return False

    def stop(self):
        self.currentFrame = QVideoFrame()
        self.targetRect = QRect()
        QAbstractVideoSurface.stop(self)
        self.widget.update()

    def present(self, frame):
        if self.surfaceFormat().pixelFormat() != frame.pixelFormat() or self.surfaceFormat().frameSize() != frame.size():
            self.setError(QAbstractVideoSurface.IncorrectFormatError)
            self.stop()
            return False
        else:
            self.currentFrame = frame
            self.widget.repaint(self.targetRect)
            return True

    def videoRect(self):
        return self.targetRect

    def updateVideoRect(self):
        origin = self.surfaceFormat().sizeHint()
        scaled = origin.scaled(self.widget.size().boundedTo(origin), Qt.KeepAspectRatio)
        self.targetRect = QRect(QPoint(0, 0), scaled);
        self.targetRect.moveCenter(self.widget.rect().center())
        if not origin.isEmpty():
            self.ratio = scaled.width() / origin.width()
            self.refPoint = self.targetRect.topLeft()

    def translatedAndScaled(self,rect):
        topLeft = (rect.topLeft()*self.ratio) + self.refPoint
        bottomRight = (rect.bottomRight()*self.ratio)+ self.refPoint
        return QRect(topLeft,bottomRight)

    def paint(self, painter):
        try:
            if self.currentFrame.map(QAbstractVideoBuffer.ReadOnly):
                oldTransform = painter.transform()

            if (self.surfaceFormat().scanLineDirection() ==
                    QVideoSurfaceFormat.BottomToTop):
                painter.scale(1, -1);
                painter.translate(0, -self.widget.height())

            image = QImage(self.currentFrame.bits(),
                           self.currentFrame.width(),
                           self.currentFrame.height(),
                           self.currentFrame.bytesPerLine(),
                           self.imageFormat
                           )
            # print(self.parent.mediaPlayer.position())
            self.i+=1
            frame_id = int(self.parent.mediaPlayer.position()/self.t_const)
            painter.drawImage(self.targetRect, image, self.sourceRect)
            bnd = self.output[frame_id]
            if bnd :
                #draw bound
                top, left, bottom, right = bnd['area'].bbox
                area_box = QRect(QPoint(left,top), QPoint(right,bottom))
                scale_box = self.translatedAndScaled(area_box)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(Qt.blue)
                painter.drawRect(scale_box)
                for cell in bnd['cells']:
                    # top, left, bottom, right = cell.bbox
                    left, top, right, bottom = cell
                    cell_box = QRect(QPoint(left, top), QPoint(right, bottom))
                    cell_box = self.translatedAndScaled(cell_box)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(Qt.green)
                    painter.drawRect(cell_box)

            painter.setTransform(oldTransform)

            self.currentFrame.unmap()
        except:
            pass


class VideoWidget(QWidget):

    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)

        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        palette = self.palette()
        palette.setColor(QPalette.Background, Qt.black)
        self.setPalette(palette)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        self.surface = VideoWidgetSurface(self, parent)

    def videoSurface(self):
        return self.surface

    def setOutput(self, output, t_const):
        self.surface.output = output
        self.surface.t_const = t_const

    def closeEvent(self, event):
        del self.surface

    def sizeHint(self):
        return self.surface.surfaceFormat().sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        if (self.surface.isActive()):
            videoRect = self.surface.videoRect()
            videoRegion = QRegion(videoRect)
            if not videoRect.contains(event.rect()):
                region = event.region()
                region.xored(videoRegion)
                brush = self.palette().brush(QPalette.Background)
                for rect in region.rects():
                    painter.fillRect(rect, brush)
            self.surface.paint(painter)
        else:
            painter.fillRect(event.rect(), self.palette().window())

    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)
        self.surface.updateVideoRect()