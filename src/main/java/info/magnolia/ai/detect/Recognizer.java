package info.magnolia.ai.detect;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Recognizer {

    private final NativeImageLoader imageLoader = new NativeImageLoader(224, 224, 3);
    private final VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();

    private final DecimalFormat numberFormat = new DecimalFormat("#.####");

    private final ComputationGraph network;
    private final List<String> labels;

    public Recognizer(ComputationGraph network, List<String> labels) {
        this.network = network;
        this.labels = labels;
    }

    public void process(File file) throws IOException {
        INDArray matrix = imageLoader.asMatrix(file);
        preProcessor.transform(matrix);
        INDArray output = network.outputSingle(matrix);
        System.out.println("*** Image " + file.getName() + " looks like:");
        printTopLabels(output);
    }

    private void printTopLabels(INDArray output) {
        Map<String, Float> scores = new HashMap<>();
        for (int i = 0; i < labels.size(); i++)
            scores.put(labels.get(i), output.getFloat(i));
        scores.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue())) // large ones first
                .limit(3)
                .forEachOrdered(e -> System.out.println(e.getKey() + ":\t " + numberFormat.format(e.getValue())));
    }
}
