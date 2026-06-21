import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet.heat'

// Google-Maps-style traffic ramp — NO blue, transparent where no traffic.
const GRADIENT = {
  0.20: '#86efac', // very low   (light green)
  0.40: '#fde047', // moderate   (yellow)
  0.58: '#f9ab00', // high       (amber)
  0.74: '#ea4335', // very heavy (red)
  0.88: '#c5221f', // congestion (deep red)
  1.0:  '#7f1d1d', // severe jam (dark red)
}

function radiusForZoom(z) {
  if (z <= 10) return 17
  if (z >= 15) return 40
  return 17 + (z - 10) * 4.6
}

function HeatLayer({ points }) {
  const map = useMap()
  const layerRef = useRef(null)
  useEffect(() => {
    const r = radiusForZoom(map.getZoom())
    const layer = L.heatLayer([], {
      radius: r, blur: r * 0.8, max: 0.9, minOpacity: 0.04,
      maxZoom: 17, gradient: GRADIENT,
    }).addTo(map)
    layerRef.current = layer
    const onZoom = () => {
      const rr = radiusForZoom(map.getZoom())
      layer.setOptions({ radius: rr, blur: rr * 0.8 })
    }
    map.on('zoomend', onZoom)
    return () => { map.off('zoomend', onZoom); map.removeLayer(layer) }
  }, [map])

  useEffect(() => {
    if (!layerRef.current || !points) return
    const pts = points.map((p) => [
      p.lat, p.lon, Math.max(0.25, Math.min(p.impact / 5, 1.5)),
    ])
    layerRef.current.setLatLngs(pts)
  }, [points])
  return null
}

// expose the map instance to the parent (for custom zoom buttons)
function MapBridge({ onReady }) {
  const map = useMap()
  useEffect(() => { onReady(map) }, [map])
  return null
}

export default function HeatMap({ points, onReady }) {
  return (
    <MapContainer center={[12.97, 77.59]} zoom={11} zoomControl={false}
      scrollWheelZoom={true} preferCanvas={true} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; OpenStreetMap, &copy; CARTO'
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
      />
      <HeatLayer points={points} />
      <MapBridge onReady={onReady} />
    </MapContainer>
  )
}
