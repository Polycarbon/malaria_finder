from PyQt5.QtCore import QRect, QPoint, Qt, QRectF, QPointF
from PyQt5.QtGui import QImage, QPalette, QPainter, QRegion, QPolygonF
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
        if handleType == QAbstractVideoBuffer.NoHandle:
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

    def transformed(self, obj):
        if type(obj) is QPolygonF:
            tranformed = QPolygonF()
            for i in range(obj.size()):
                tranformed << (obj.at(i) * self.ratio) + self.refPoint
            return tranformed
        else:
            top_left = (obj.topLeft() * self.ratio) + self.refPoint
            bottom_right = (obj.bottomRight() * self.ratio) + self.refPoint
            return QRectF(top_left, bottom_right)

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
            self.i += 1
            frame_id = int(self.parent.mediaPlayer.position() / self.t_const)
            painter.drawImage(self.targetRect, image, self.sourceRect)
            objects = self.output[frame_id]['cells']
            for o_id, cell in objects.items():
                # top, left, bottom, right = cell.bbox
                cell_box = self.transformed(cell)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(Qt.red)
                if cell.isCounted():
                    painter.setPen(Qt.green)
                    painter.drawText(cell_box.bottomRight(), "id {}".format(cell.getCountId()))
                painter.drawPoint(cell_box.center())
                painter.drawRect(cell_box)
            area = self.output[frame_id]['area']
            if area:
                area = self.transformed(area)
                painter.setPen(Qt.blue)
                painter.drawPolygon(area)
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
