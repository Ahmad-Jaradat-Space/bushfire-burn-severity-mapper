<!DOCTYPE qgis>
<!-- QGIS layer style for the 4-class burn-severity rasters / polygons.
     Palette matches src/viz/theme.py so QGIS, the notebook and the README
     all render severity in the same colours. Load via Layer Properties →
     Symbology → Style → Load Style. -->
<qgis version="3.34" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
        <paletteEntry value="0" color="#5C8A6B" label="Unburnt" alpha="255"/>
        <paletteEntry value="1" color="#D8A256" label="Low–Moderate" alpha="255"/>
        <paletteEntry value="2" color="#C5683B" label="High" alpha="255"/>
        <paletteEntry value="3" color="#7F1F1F" label="Very High" alpha="255"/>
        <paletteEntry value="255" color="#000000" label="No data" alpha="0"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation saturation="0" grayscaleMode="0"/>
  </pipe>
</qgis>
