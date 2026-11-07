/* Batch adapter for the UNMODIFIED upstream Shotwell 0.30.14 colour engine.
 * This adapter does not implement or replace the enhancement algorithm.
 * Inputs: GUI-equivalent 360px analysis PNG and a 16x16 RGB palette PNG.
 * No GTK UI or photo library/database is opened.
 */

// KeyValueMap's unused red-eye helpers require only this coordinate type.
// This build-only shim avoids installing GTK development packages; no colour
// calculation uses it. GdkPixbuf itself comes from the system's real library.
namespace Gdk {
    public struct Point {
        public int x;
        public int y;
    }
}

int main(string[] args) {
    if (args.length != 5) {
        stderr.printf("Usage: engine analysis.png palette.png enhanced_palette.png parameters.json\n");
        return 2;
    }
    try {
        var analysis = new Gdk.Pixbuf.from_file(args[1]);
        var palette = new Gdk.Pixbuf.from_file(args[2]);
        if (palette.width != 16 || palette.height != 16 || palette.n_channels != 3)
            return 3;
        var bundle = AutoEnhance.create_auto_enhance_adjustments(analysis);
        var transformer = bundle.generate_transformer();
        var result = palette.copy();
        transformer.transform_to_other_pixbuf(palette, result, null, 1);
        result.save(args[3], "png");
        var expansion = (ExpansionTransformation) bundle.get_transformation(
            PixelTransformationType.TONE_EXPANSION);
        var shadows = (ShadowDetailTransformation) bundle.get_transformation(
            PixelTransformationType.SHADOWS);
        var order = new StringBuilder();
        foreach (var transformation in bundle.get_transformations()) {
            if (order.len > 0) order.append(",");
            order.append_printf("%d", (int) transformation.get_transformation_type());
        }
        var report = "{\"engine\":\"unmodified upstream Shotwell 0.30.14\", " +
            "\"analysis_dimensions\":[%d,%d], \"black_point\":%d, " +
            "\"white_point\":%d, \"shadow_parameter\":%.9g, " +
            "\"actual_Gee_transformation_order\":[%s]}\n";
        FileUtils.set_contents(args[4], report.printf(analysis.width, analysis.height,
            expansion.get_black_point(), expansion.get_white_point(),
            shadows.get_parameter(), order.str));
        return 0;
    } catch (Error err) {
        stderr.printf("%s\n", err.message);
        return 1;
    }
}
