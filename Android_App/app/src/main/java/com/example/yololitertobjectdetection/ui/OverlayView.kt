package com.example.yololitertobjectdetection.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.yololitertobjectdetection.BoundingBox

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var inferenceTime = 0L
    private var fps = 0f

    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()
    private val statsPaint = Paint()
    private val countPaint = Paint()

    private var bounds = Rect()
    private val colorMap = mutableMapOf<String, Int>()

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        inferenceTime = 0L
        fps = 0f
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.WHITE
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 42f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 42f

        // Small text in top-left corner showing inference time and FPS
        statsPaint.color = Color.WHITE
        statsPaint.style = Paint.Style.FILL
        statsPaint.textSize = 36f
        statsPaint.setShadowLayer(4f, 0f, 0f, Color.BLACK)

        // Detection count badge
        countPaint.color = Color.WHITE
        countPaint.style = Paint.Style.FILL
        countPaint.textSize = 36f
        countPaint.setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach { boundingBox ->
            val color = getColorForLabel(boundingBox.clsName)

            boxPaint.color = color
            boxPaint.strokeWidth = 8F
            boxPaint.style = Paint.Style.STROKE

            val left   = boundingBox.x1 * width
            val top    = boundingBox.y1 * height
            val right  = boundingBox.x2 * width
            val bottom = boundingBox.y2 * height

            canvas.drawRoundRect(left, top, right, bottom, 16f, 16f, boxPaint)

            val drawableText =
                "${boundingBox.clsName} ${Math.round(boundingBox.cnf * 100.0) / 100.0}"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth  = bounds.width()
            val textHeight = bounds.height()

            val textBackgroundRect = RectF(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING
            )
            textBackgroundPaint.color = color
            canvas.drawRoundRect(textBackgroundRect, 8f, 8f, textBackgroundPaint)
            canvas.drawText(drawableText, left, top + textHeight, textPaint)

            // Draw a thin confidence bar along the bottom edge of the bounding box
            val barWidth = (right - left) * boundingBox.cnf
            boxPaint.style = Paint.Style.FILL
            boxPaint.alpha = 160
            canvas.drawRect(left, bottom - CONF_BAR_HEIGHT, left + barWidth, bottom, boxPaint)
            boxPaint.alpha = 255
        }

        // Stats overlay in the top-left corner
        if (inferenceTime > 0) {
            val statsText = "${inferenceTime}ms  ${fps.toInt()} FPS"
            canvas.drawText(statsText, STATS_PADDING, STATS_PADDING + 36f, statsPaint)
        }

        // Detection count in top-right corner
        if (results.isNotEmpty()) {
            val countText = "${results.size} detected"
            countPaint.getTextBounds(countText, 0, countText.length, bounds)
            canvas.drawText(countText, width - bounds.width() - STATS_PADDING, STATS_PADDING + 36f, countPaint)
        }
    }

    private fun getColorForLabel(label: String): Int {
        return colorMap.getOrPut(label) {
            Color.rgb((0..255).random(), (0..255).random(), (0..255).random())
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    fun setInferenceTime(timeMs: Long) {
        inferenceTime = timeMs
        fps = if (timeMs > 0) 1000f / timeMs else 0f
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
        private const val CONF_BAR_HEIGHT = 6f
        private const val STATS_PADDING = 16f
    }
}
