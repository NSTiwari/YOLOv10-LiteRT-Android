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
    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()

    private var bounds = Rect()
    private val colorMap = mutableMapOf<String, Int>()

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
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
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach { boundingBox ->
            // Get or create a color for this label
            val color = getColorForLabel(boundingBox.clsName)
            boxPaint.color = color
            boxPaint.strokeWidth = 8F
            boxPaint.style = Paint.Style.STROKE

            val left = boundingBox.x1 * width
            val top = boundingBox.y1 * height
            val right = boundingBox.x2 * width
            val bottom = boundingBox.y2 * height

            canvas.drawRoundRect(left, top, right, bottom, 16f, 16f, boxPaint)

            val drawableText = "${boundingBox.clsName} ${Math.round(boundingBox.cnf * 100.0) / 100.0}"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            val textBackgroundRect = RectF(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING
            )
            textBackgroundPaint.color = color // Set background color same as bounding box
            canvas.drawRoundRect(textBackgroundRect, 8f, 8f, textBackgroundPaint)

            canvas.drawText(drawableText, left, top + textHeight, textPaint)
        }
    }

    private fun getColorForLabel(label: String): Int {
        return colorMap.getOrPut(label) {
            // Generate a random color or you can use a predefined set of colors
            Color.rgb((0..255).random(), (0..255).random(), (0..255).random())
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}