package info.magnolia.ai;

import static java.util.stream.Collectors.toList;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.sf.extjwnl.data.IndexWord;

public class NetworkManager {

    private static final boolean STATS_ON = false;
    private static final int STORE_FREQUENCY = 20;

    private static final Logger log = LoggerFactory.getLogger(NetworkManager.class);

    private final List<IndexWord> labels;
    private final ComputationGraph network;
    private final TransferLearningHelper transferHelper;
    private final LocalDateTime now = LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS);
    private final File persistenceFile = new File("custom-images-trained-network_" + now.format(DateTimeFormatter.ISO_DATE_TIME));
    private final File labelsFile = new File("custom-images-labels_" + now.format(DateTimeFormatter.ISO_DATE_TIME));
    private final StatsStorage statsStorage = new InMemoryStatsStorage();

    public NetworkManager(List<IndexWord> labels) {
        this.labels = labels;
        network = buildNetwork();
        transferHelper = new TransferLearningHelper(network);

        if (STATS_ON) UIServer.getInstance().attach(statsStorage);

        // GC less often
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
    }

    public TransferLearningHelper getTransferHelper() {
        return transferHelper;
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
                .updater(new Nesterovs(5e-4, 0.5))
                .build();
        ComputationGraph transferGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("fc2") // freeze this and below
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                                .nIn(4096).nOut(labels.size())
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + labels.size()))))
                                .activation(Activation.TANH)
                                .build(), "fc2")
                .build();

        log.info("Transfer model:\n{}", transferGraph.summary());

        return transferGraph;
    }

    public void train(DataSetIterator trainIterator, DataSetIterator testIterator, int epochs) {
        if (STATS_ON) transferHelper.unfrozenGraph().setListeners(new StatsListener(statsStorage, 100));

        log.info("Going to featurize images...");
        // load ahead of time rather than lazily to avoid issues with multiple workspaces (and cache featurized already)
        preLoad(trainIterator);
        preLoad(testIterator);

        List<String> labelStrings = labels.stream().map(IndexWord::getLemma).collect(toList());

        Evaluation evalBefore = transferHelper.unfrozenGraph().evaluate(testIterator, labelStrings);
        log.info(evalBefore.stats(false, false));
        testIterator.reset();

        for (int i = 0; i < epochs; i++) {
            log.info("*** Starting training epoch {}", i);

            transferHelper.fitFeaturized(trainIterator);
            trainIterator.reset();

            log.info("Epoch done, starting evaluation...");

            Evaluation eval = transferHelper.unfrozenGraph().evaluate(testIterator, labelStrings, 3);
            log.info(eval.stats(false, false));
            testIterator.reset();

            if (i > 0 && i % STORE_FREQUENCY == 0) {
                log.info("Going to store results...");
                store();
            }
        }

        if ((epochs - 1) % STORE_FREQUENCY != 0) {
            log.info("Training complete, storing one last time...");
            store();
        }

        log.info("DONE");
    }

    public void store() {
        try {
            ModelSerializer.writeModel(network, persistenceFile, true);
            log.info("Stored trained network to: {}", persistenceFile.getAbsolutePath());

            List<String> labelStrings = labels.stream().map(IndexWord::getLemma).collect(Collectors.toList());
            Files.write(labelsFile.toPath(), labelStrings);
            log.info("Stored labels to: {}", labelsFile.getAbsolutePath());
        } catch (IOException e) {
            log.error("Failed to store outcome", e);
        }
    }

    private void preLoad(DataSetIterator dataSetIterator) {
        while (dataSetIterator.hasNext()) dataSetIterator.next();
        dataSetIterator.reset();
    }
}
