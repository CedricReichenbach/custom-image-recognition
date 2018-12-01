package info.magnolia.ai;

import net.sf.extjwnl.data.IndexWord;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.Predicate;

import static java.util.stream.Collectors.toMap;

public class TrainingOrganizer {

    private static final int EPOCHS = 1;

    private final ImageIndex imageIndex;
    private final NetworkManager networkManager;

    public TrainingOrganizer() {
        this.imageIndex = new ImageIndex();
        this.networkManager = new NetworkManager(imageIndex.getLabels());

        System.out.println(imageIndex);
    }

    @NotNull
    private BaseDatasetIterator buildIterator(Predicate<String> filter) {
        Map<String, Set<IndexWord>> images = this.imageIndex.getImages().entrySet().stream()
                .filter(entry -> filter.test(entry.getKey()))
                .collect(toMap(Entry::getKey, Entry::getValue));
        DataSetFetcher fetcher = new ImageNetDataFetcher(images, imageIndex.getLabels());
        return new BaseDatasetIterator(5, images.size(), fetcher);
    }

    /**
     * Deterministically select certain images for evaluation/testing group as opposed to training (based on hash).
     */
    private boolean useForEval(String imageUrl) {
        // pick roughly 1 out of 4 items
        return (imageUrl.hashCode() & 0b11) == 0;
    }

    public void train() {
        BaseDatasetIterator evalIterator = buildIterator(this::useForEval);
        BaseDatasetIterator trainIterator = buildIterator(url -> !this.useForEval(url));

        networkManager.train(trainIterator, evalIterator, EPOCHS);
    }
}
