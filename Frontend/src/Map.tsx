import { useEffect, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay, Rectangle } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import 'leaflet/dist/leaflet.css';

interface MapProperties {
    type: "pre" | "post";
    boundingBoxes?: BoundingBox[];
    damageFilter?: {
        noDamage: boolean;
        minorDamage: boolean;
        severeDamage: boolean;
    }
}

export type BoundingBox = {
    building_id: string;
    subtype: string;
    bbox: [number, number, number, number];
}

type GeotransformEntry = [[number, number, number, number, number, number], string]
type GeotransformJSON = Record<string, GeotransformEntry>
type Geotransform = [number, number, number, number, number, number];

const S3_IMGS = "https://amzn-santa-rosa-wildfire-images.s3.us-east-1.amazonaws.com/datatsetCapstone"
const IMG_SIZE = 1024

//This is where the maps images are added. Its getting the edges of each image and calculating it.
function getBounds(geotransform: [number, number, number, number, number, number]): LatLngBoundsExpression {

    const padding = 0.00

    const [long, pixelW, , lat, , pixelH] = geotransform
    const north = lat
    const south = lat + (pixelH * (IMG_SIZE)) + padding
    const west = long
    const east = long + (pixelW * (IMG_SIZE)) - padding
    return [[south, west], [north, east]]
}

function pixeltoLatLng(px: number, py: number, geotransform: Geotransform): [number, number] {
    const [originLng, pixelW, , originLat, , pixelH] = geotransform
    const lng = originLng + px * pixelW
    const lat = originLat + py * pixelH
    return[lat, lng]
}

function subtypeDamageFilter(subtype: string, damageFilter: MapProperties["damageFilter"]): boolean {
    if (!damageFilter) return true;
    if(subtype == "no-damage") return damageFilter.noDamage;
    if(subtype == "minor-damage") return damageFilter.minorDamage;
    return damageFilter.severeDamage;
}

function subtypeColor(subtype: string): string {
    if (subtype == "no-damage") return "#22c55e";
    if (subtype == "minor-damage") return "#f59e0b";
    if (subtype == "un-classified") return "#9ca3af";
    return "#ef4444";
}

export default function Map({type, boundingBoxes = [], damageFilter}: MapProperties){
    const [overlays, setOverlay] = useState<{ 
        url: string; 
        bounds: LatLngBoundsExpression;
        geotransform: Geotransform;
        imageID: string;
    } []>([])

    useEffect(() => {
        fetch("/public/santa_rose_geotransform.json")
        .then (res => res.json())
        .then((data: GeotransformJSON) => {
            const suffix = type === "pre" ? "pre_disaster" : "post_disaster"

            const filtered = Object.entries(data)
                .filter(([filename]) => filename.includes(suffix))
                .map(([filename, [geotransform]]) => {
                    const match = filename.match(/santa-rosa-wildfire_(\d+)_/)
                    const imageID = match ? `santa-rosa-${match[1]}`: filename.replace(`_${suffix}.png`, "")
                    return {
                        url: `${S3_IMGS}/${filename}`,
                        bounds: getBounds(geotransform),
                        geotransform,
                        imageID
                }})
            setOverlay(filtered)
        })
    }, [type])

    return (
        <MapContainer
            center = {[38.43528784609693, -122.71303705013497]}
            zoom = {13}
            style = {{height: "100%", width: "100%"}}
        >
            <TileLayer
                url = "https://api.maptiler.com/maps/satellite-v4/{z}/{x}/{y}.jpg?key=g5cvX3fWQLMVoSF0kway"
                tileSize = {512}
                zoomOffset= {-1}
                opacity={0.5}
                attribution="© MapTiler © OpenStreetMap contributors"
            />
            {overlays.map(({url, bounds}) => (
                <ImageOverlay
                    key={url}
                    url={url}
                    bounds={bounds}
                    opacity={1.0}
                />
            ))}
            {boundingBoxes
            .filter(box => subtypeDamageFilter(box.subtype, damageFilter))
            .map(({building_id, subtype, bbox}) => {
                const imageIDMatch = building_id.match(/^(santa-rosa-\d+)/)
                const imageID = imageIDMatch ? imageIDMatch[1] : null
                const tileOverlay = imageID ? overlays.find(o => o.imageID == imageID) : overlays[0]
                
                if(!tileOverlay) return null

                const [minX, minY, maxX, maxY] = bbox
                const sw = pixeltoLatLng(minX, maxY, tileOverlay.geotransform)
                const ne = pixeltoLatLng(maxX, minY, tileOverlay.geotransform)

                return (
                    <Rectangle
                        key = {building_id}
                        bounds = {[sw, ne]}
                        pathOptions={{
                            color: subtypeColor(subtype),
                            weight: 1.5,
                            fill: true,
                            fillOpacity: 0.15,
                            dashArray: "4 2"
                        }}
                    />
                )
            })}
        </MapContainer>
    )
}