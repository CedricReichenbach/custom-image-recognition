package info.magnolia.ai;

import static java.util.stream.Collectors.toMap;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.Predicate;

import net.sf.extjwnl.data.Synset;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainingOrganizer {

    private static final Logger log = LoggerFactory.getLogger(TrainingOrganizer.class);

    private static final int EPOCHS = 100;

    private final ImageIndex imageIndex;
    private final NetworkManager networkManager;

    public TrainingOrganizer() {
        this.imageIndex = new ImageIndex();
        this.networkManager = new NetworkManager(imageIndex.getLabels());

        log.info("Image index: {}", imageIndex);
    }

    @NotNull
    private DataSetIterator buildIterator(Predicate<String> filter) {
        Map<String, Set<Synset>> images = this.imageIndex.getImages().entrySet().stream()
                .filter(entry -> filter.test(entry.getKey()))
                .collect(toMap(Entry::getKey, Entry::getValue));

        // FIXME: Why are some words in images values not in labels?

        DataSetFetcher fetcher = new FeaturizedFetcher(images, imageIndex.getLabels(), networkManager.getTransferHelper());
        return new CachingDataSetIterator(new BaseDatasetIterator(50, images.size(), fetcher), new InMemoryDataSetCache());
    }

    /**
     * Deterministically select certain images for evaluation/testing group as opposed to training (based on hash).
     */
    private boolean useForEval(String imageUrl) {
        // pick roughly 1 out of 4 items
        return (imageUrl.hashCode() & 0b11) == 0;
    }

    public void train() {
        DataSetIterator trainIterator = buildIterator(url -> !this.useForEval(url));
        DataSetIterator evalIterator = buildIterator(this::useForEval);

        networkManager.train(trainIterator, evalIterator, EPOCHS);
    }
}
