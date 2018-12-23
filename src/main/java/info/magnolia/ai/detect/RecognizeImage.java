package info.magnolia.ai.detect;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.List;

public class RecognizeImage {

    private static final File MODEL_FILE = new File("custom-images-trained-network_2018-12-22T17:40:02");
    private static final File LABELS_FILE = new File("custom-images-labels_2018-12-22T17:40:02");

    public static void main(String[] args) throws IOException, URISyntaxException {
        System.out.println("Loading neural network from " + MODEL_FILE.getName());
        ComputationGraph network = ModelSerializer.restoreComputationGraph(MODEL_FILE);
        List<String> labels = Files.readAllLines(LABELS_FILE.toPath());
        Recognizer recognizer = new Recognizer(network, labels);

        File directory = new File(RecognizeImage.class.getResource(".").toURI());
        for (File file : directory.listFiles()) {
            String filename = file.getName().toLowerCase();
            if (filename.endsWith(".jpg") || filename.endsWith(".jpeg") || filename.endsWith(".png"))
                recognizer.process(file);
        }
    }
}
