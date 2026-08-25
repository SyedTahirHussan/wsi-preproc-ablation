/**
 * Load a tissue mask and a tile grid exported by `wsi-ablation qupath-export`
 * into the currently open slide.
 *
 * QuPath 0.4 or later. Run from Automate > Script editor with the slide open.
 *
 * The GeoJSON is written in level-0 image coordinates, which is the coordinate
 * space QuPath uses, so nothing here rescales anything. If annotations land in
 * the top-left corner of the slide, the exporter wrote overview coordinates and
 * the bug is on the Python side, not here.
 */

import qupath.lib.io.PathIO
import qupath.lib.scripting.QP

def slideName = QP.getCurrentImageData().getServer().getMetadata().getName()
def stem = slideName.replaceFirst(/\.[^.]+$/, '')
def dir = new File(buildFilePath(PROJECT_BASE_DIR, 'pipeline'))

['tissue', 'tiles'].each { kind ->
    def file = new File(dir, "${stem}-${kind}.geojson")
    if (!file.exists()) {
        print "no ${kind} file at ${file}"
        return
    }
    def objects = PathIO.readObjects(file)
    QP.addObjects(objects)
    print "added ${objects.size()} ${kind} annotations"
}

QP.fireHierarchyUpdate()
