package info.magnolia.ai;

import net.sf.extjwnl.data.IndexWord;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;

public class NetworkManager {

    private final List<IndexWord> labels;
    private final ComputationGraph network;

    public NetworkManager(List<IndexWord> labels) {
        this.labels = labels;
        network = buildNetwork();

        initStats();
    }

    private void initStats() {
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        UIServer.getInstance().attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage));
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
                                .nIn(4096).nOut(labels.size())
                                .weightInit(WeightInit.ZERO)
                                .activation(Activation.SIGMOID)
                                .build(), "fc2")
                .build();

        System.out.println("Transfer model:");
        System.out.println(transferGraph.summary());

        return pretrainedNet;
    }

    public void train(DataSetIterator trainIterator, DataSetIterator testIterator, int epochs) {
        network.evaluate(testIterator);
        testIterator.reset();

        for (int i = 0; i < epochs; i++) {
            network.fit(trainIterator);
            trainIterator.reset();

            network.evaluate(testIterator);
            testIterator.reset();
        }
    }
}
