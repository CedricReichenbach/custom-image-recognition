package info.magnolia.ai;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkManager {

    private final String[] labels;
    private final ComputationGraph network;

    public NetworkManager(String[] labels) {
        this.labels = labels;
        network = buildNetwork();
    }

    private ComputationGraph buildNetwork() {
        ComputationGraph pretrainedNet;
        try {
            pretrainedNet = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load pre-trained network", e);
        }

        final FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(1e-4, 0.1))
                .build();
        ComputationGraph transferGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("fc2") // freeze this and below
//                .removeVertexAndConnections("predictions") // XXX: Maybe this?
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                .nIn(4096).nOut(labels.length) // santa hat or not santa hat
                                .weightInit(WeightInit.ZERO)
                                .activation(Activation.SIGMOID)
                                .build(), "fc2")
                .build();

        System.out.println("Transfer model:");
        System.out.println(transferGraph.summary());

        return pretrainedNet;
    }
}
