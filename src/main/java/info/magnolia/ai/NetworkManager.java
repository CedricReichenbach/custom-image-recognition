package info.magnolia.ai;

import static java.util.stream.Collectors.toList;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import net.sf.extjwnl.data.IndexWord;

public class NetworkManager {

    private final List<IndexWord> labels;
    private final ComputationGraph network;
    private final File persistenceFile = new File("custom-images-trained-network_" + LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME));
    private final File labelsFile = new File("custom-images-labels_" + LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME));
    private final InMemoryStatsStorage statsStorage;

    public NetworkManager(List<IndexWord> labels) {
        this.labels = labels;
        network = buildNetwork();

        statsStorage = new InMemoryStatsStorage();
        UIServer.getInstance().attach(statsStorage);
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
                .updater(new Nesterovs(1e-4, 0.5))
                .build();
        ComputationGraph transferGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("fc2") // freeze this and below
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(labels.size())
                                .weightInit(WeightInit.ZERO)
                                .activation(Activation.SOFTMAX)
                                .build(), "fc2")
                .build();

        System.out.println("Transfer model:");
        System.out.println(transferGraph.summary());

        return transferGraph;
    }

    public void train(DataSetIterator trainIterator, DataSetIterator testIterator, int epochs) {
        TransferLearningHelper transferHelper = new TransferLearningHelper(network);
        transferHelper.unfrozenGraph().setListeners(new StatsListener(statsStorage, 1));

        System.out.println("Going to featurize images...");
        // TODO: Cache featurized datasets?
        DataSetIterator featurizedTrain = featurize(trainIterator, transferHelper);
        DataSetIterator featurizedTest = featurize(testIterator, transferHelper);

        List<String> labelStrings = labels.stream().map(IndexWord::getLemma).collect(toList());

        Evaluation evalBefore = transferHelper.unfrozenGraph().evaluate(featurizedTest, labelStrings);
        System.out.println(evalBefore.stats(false, false));
        featurizedTest.reset();

        for (int i = 0; i < epochs; i++) {
            System.out.println("Starting training epoch " + i);

            transferHelper.fitFeaturized(featurizedTrain);
            featurizedTrain.reset();

            Evaluation eval = transferHelper.unfrozenGraph().evaluate(featurizedTest, labelStrings);
            System.out.println(eval.stats(false, false));
            featurizedTest.reset();

            store();
        }

        System.out.println("Training complete");
    }

    public void store() {
        try {
            ModelSerializer.writeModel(network, persistenceFile, true);
            System.out.println("Stored trained network to: " + persistenceFile.getAbsolutePath());

            List<String> labelStrings = labels.stream().map(IndexWord::getLemma).collect(Collectors.toList());
            Files.write(labelsFile.toPath(), labelStrings);
            System.out.println("Stored labels to: " + labelsFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private DataSetIterator featurize(DataSetIterator dataSetIterator, TransferLearningHelper transferHelper) {
        // featurize ahead of time rather than lazily to avoid issues with multiple workspaces
        List<DataSet> featurizeds = new LinkedList<>();
        while (dataSetIterator.hasNext()) {
            DataSet dataSet = dataSetIterator.next();
            featurizeds.add(transferHelper.featurize(dataSet));
        }
        return new CachingDataSetIterator(
                new ListDataSetIterator<>(featurizeds, dataSetIterator.batch()),
                new InMemoryDataSetCache()
        );
    }
}
