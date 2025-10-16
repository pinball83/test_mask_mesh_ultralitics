#version 300 es
precision highp float;

#include <flutter/runtime_effect.glsl>

uniform sampler2D uMaskTexture;
uniform sampler2D uSourceTexture;
uniform sampler2D uMustacheTexture;

// Display and image uniforms
uniform float uWidth;
uniform float uHeight;
uniform float uMode;
uniform float uFlipX;
uniform float uFlipY;

// Image metadata uniforms
uniform float uImageWidth;
uniform float uImageHeight;
uniform float uRotation; // 0=0°, 1=90°, 2=180°, 3=270°

// Face detection uniforms
uniform float uFaceDetected; // 0=no face, 1=face detected
uniform float uFaceBoundingLeft;
uniform float uFaceBoundingTop;
uniform float uFaceBoundingRight;
uniform float uFaceBoundingBottom;

// Face landmark uniforms (nose bridge points)
uniform float uNoseBridgeCount;
uniform float uNoseBridgeStartX;
uniform float uNoseBridgeStartY;
uniform float uNoseBridgeEndX;
uniform float uNoseBridgeEndY;

// Face landmark uniforms (upper lip points)
uniform float uUpperLipCount;
uniform float uUpperLipCenterX;
uniform float uUpperLipCenterY;

// Mustache appearance uniforms
uniform float uMustacheScale;
uniform float uMustacheAlpha;

out vec4 fragColor;

// Platform detection
bool isIOS() {
    // This is a simplified check - in practice you'd pass this as a uniform
    return false; // For now, assume Android - pass this as uniform if needed
}

// Coordinate transformation functions (ported from Dart)
float translateX(float x, float canvasWidth, float canvasHeight, float imageWidth, float imageHeight, int rotation, int uFlipX, bool isDelta) {
    float scaleX = canvasWidth / imageWidth;

    if (isDelta) {
        return x * scaleX;
    }

    if (rotation == 1) { // 90 degrees
        return x * canvasWidth / (isIOS() ? imageWidth : imageHeight);
    } else if (rotation == 3) { // 270 degrees
        return canvasWidth - x * canvasWidth / (isIOS() ? imageWidth : imageHeight);
    } else { // 0 or 180 degrees
        if (uFlipX == 0) { // back camera
            return x * canvasWidth / imageWidth;
        } else { // front camera
            return canvasWidth - x * canvasWidth / imageWidth;
        }
    }
}

float translateY(float y, float canvasWidth, float canvasHeight, float imageWidth, float imageHeight, int rotation, int uFlipX, bool isDelta) {
    float scaleY = canvasHeight / imageHeight;

    if (isDelta) {
        return y * scaleY;
    }

    if (rotation == 1 || rotation == 3) { // 90 or 270 degrees
        return y * canvasHeight / (isIOS() ? imageHeight : imageWidth);
    } else { // 0 or 180 degrees
        return y * canvasHeight / imageHeight;
    }
}

// Calculate mustache position based on face landmarks
vec2 calculateMustachePosition() {
    // Default position (center)
    vec2 mustachePos = vec2(0.5, 0.6);

    if (uFaceDetected < 0.5) {
        return mustachePos; // No face detected, return default
    }

    // Use face landmarks for precise positioning
    if (uNoseBridgeCount > 0.5 && uUpperLipCount > 0.5) {
        // Position mustache between nose bottom and upper lip (30% from nose to lip)
        float mustacheX = uNoseBridgeEndX;
        float mustacheY = uNoseBridgeEndY + (uUpperLipCenterY - uNoseBridgeEndY) * 0.3;

        // Transform coordinates from image space to display space
        int rotation = int(uRotation);
        int uFlipX = int(uFlipX);

        float transformedX = translateX(
                mustacheX,
                uWidth, uHeight,
                uImageWidth, uImageHeight,
                rotation, uFlipX,
                false
            );
        float transformedY = translateY(
                mustacheY,
                uWidth, uHeight,
                uImageWidth, uImageHeight,
                rotation, uFlipX,
                false
            );

        // Convert to normalized UV coordinates
        mustachePos = vec2(transformedX / uWidth, transformedY / uHeight);
    } else {
        // Fallback: use face bounding box center with estimated mustache position
        float faceCenterX = (uFaceBoundingLeft + uFaceBoundingRight) * 0.5;
        float faceCenterY = (uFaceBoundingTop + uFaceBoundingBottom) * 0.5;
        float faceHeight = uFaceBoundingBottom - uFaceBoundingTop;

        // Position mustache slightly below face center
        float fallbackMustacheX = faceCenterX;
        float fallbackMustacheY = faceCenterY + faceHeight * 0.15;

        // Transform coordinates
        int rotation = int(uRotation);
        int uFlipX = int(uFlipX);

        float transformedX = translateX(
                fallbackMustacheX,
                uWidth, uHeight,
                uImageWidth, uImageHeight,
                rotation, uFlipX,
                false
            );
        float transformedY = translateY(
                fallbackMustacheY,
                uWidth, uHeight,
                uImageWidth, uImageHeight,
                rotation, uFlipX,
                false
            );

        // Convert to normalized UV coordinates
        mustachePos = vec2(transformedX / uWidth, transformedY / uHeight);
    }

    // Clamp to valid UV range
    mustachePos = clamp(mustachePos, vec2(0.0), vec2(1.0));

    return mustachePos;
}

// Calculate mustache rotation based on nose bridge angle
float calculateMustacheRotation() {
    if (uFaceDetected < 0.5 || uNoseBridgeCount < 1.5) {
        return 0.0; // No face or insufficient nose bridge points
    }

    // Calculate rotation based on nose bridge angle
    float deltaY = uNoseBridgeEndY - uNoseBridgeStartY;
    float deltaX = uNoseBridgeEndX - uNoseBridgeStartX;
    float rotation = atan(deltaY, deltaX) - (3.14159 / 2.0); // Adjust for vertical nose

    // Limit rotation to reasonable angles (-20 to +20 degrees)
    float maxRotation = 3.14159 / 9.0; // 20 degrees in radians
    rotation = clamp(rotation, -maxRotation, maxRotation);

    return rotation;
}

// Calculate dynamic mustache scale based on face size
float calculateMustacheScale() {
    if (uFaceDetected < 0.5) {
        return uMustacheScale; // Use default scale if no face
    }

    // Calculate face dimensions
    float faceWidth = uFaceBoundingRight - uFaceBoundingLeft;
    float normalizedFaceWidth = faceWidth / uImageWidth;

    // Scale mustache relative to face width
    float dynamicScale = max(0.06, min(0.20, normalizedFaceWidth * 0.5));

    return dynamicScale;
}

void main() {
    vec2 sizeVec = vec2(uWidth, uHeight);
    vec2 uv = FlutterFragCoord().xy / sizeVec;

    // Apply flipping for camera orientation
    if (uFlipX > 0.5) uv.x = 1.0 - uv.x;
    if (uFlipY > 0.5) uv.y = 1.0 - uv.y;

    vec4 sourceColor = texture(uSourceTexture, uv);

    // Background replacement with person segmentation
    float maskValue = texture(uMaskTexture, uv).r;

    if (uMode > 1.5) {
        if (maskValue < 0.5) {
            // Background area - render the background image
            fragColor = vec4(sourceColor.rgb, 1.0);
        } else {
            // Person area - render transparent so camera feed shows through
            fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        }
    } else if (uMode > 0.5) {
        if (maskValue < 0.5) {
            fragColor = vec4(sourceColor.rgb, 1.0);
        } else {
            fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        }
    } else {
        // Effect disabled - keep overlay fully transparent
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }

    // ----- GPU-optimized Mustache Overlay -----
    if (uFaceDetected > 0.5 && uMustacheAlpha > 0.01) {
        // Calculate mustache position dynamically in shader
        vec2 mustacheCenter = calculateMustachePosition();

        // Calculate dynamic scale and rotation
        float dynamicScale = calculateMustacheScale();
        float mustacheRotation = calculateMustacheRotation();

        // Apply same flipping to mustache center as UV coordinates
        if (uFlipX > 0.5) mustacheCenter.x = 1.0 - mustacheCenter.x;
        if (uFlipY > 0.5) mustacheCenter.y = 1.0 - mustacheCenter.y;

        // Account for aspect ratio differences
        float aspectRatio = uWidth / uHeight;

        // Compute local coordinates relative to mustache center
        vec2 local = uv - mustacheCenter;
        local.x *= aspectRatio;

        // Apply rotation
        float c = cos(-mustacheRotation);
        float s = sin(-mustacheRotation);
        mat2 rot = mat2(c, -s, s, c);
        vec2 localRot = rot * local;

        // Apply dynamic scaling with aspect ratio correction
        float adjustedScale = dynamicScale * aspectRatio;
        vec2 mustacheUV = vec2(localRot.x / adjustedScale, localRot.y / dynamicScale) + vec2(0.5, 0.5);

        // Sample mustache texture
        vec4 mustacheColor = vec4(0.0);
        if (mustacheUV.x >= 0.0 && mustacheUV.x <= 1.0 && mustacheUV.y >= 0.0 && mustacheUV.y <= 1.0) {
            mustacheColor = texture(uMustacheTexture, mustacheUV);
        }

        // Composite mustache with alpha blending
        float maskA = mustacheColor.a * uMustacheAlpha;
        vec3 outRgb = mix(fragColor.rgb, mustacheColor.rgb, maskA);
        float outA = max(fragColor.a, maskA);
        fragColor = vec4(outRgb, outA);
    }
}
