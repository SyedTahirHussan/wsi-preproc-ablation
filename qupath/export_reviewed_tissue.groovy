/**
 * Send a pathologist's corrected tissue outline back to the pipeline.
 *
 * Select the annotations to keep, classify them as "Tissue", then run this. The
 * file it writes is what `wsi_ablation.qupath.geojson_to_mask` reads, so a
 * corrected mask can be substituted for either detector arm and the rest of the
 * ablation runs unchanged.
 *
 * QuPath 0.4 or later.
 */

import qupath.lib.io.PathIO
import qupath.lib.scripting.QP

def annotations = QP.getAnnotationObjects().findAll {
    it.getPathClass() != null && it.getPathClass().getName() == 'Tissue'
}

if (annotations.isEmpty()) {
    print 'nothing classified as Tissue; classify the annotations first'
    return
}

def slideName = QP.getCurrentImageData().getServer().getMetadata().getName()
def stem = slideName.replaceFirst(/\.[^.]+$/, '')
def dir = new File(buildFilePath(PROJECT_BASE_DIR, 'reviewed'))
dir.mkdirs()

def out = new File(dir, "${stem}-tissue-reviewed.geojson")
PathIO.exportObjectsAsGeoJSON(out, annotations, 'FEATURE_COLLECTION')
print "wrote ${annotations.size()} reviewed annotations to ${out}"
