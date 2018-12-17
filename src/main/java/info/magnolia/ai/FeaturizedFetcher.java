package info.magnolia.ai;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import net.sf.extjwnl.data.IndexWord;

public class FeaturizedFetcher extends ImageNetDataFetcher {

    private final TransferLearningHelper transferHelper;
    private final FileSystemCache featurizedCache = new FileSystemCache("custom-image-recognition-samples_featurized");

    public FeaturizedFetcher(Map<String, Set<IndexWord>> images, List<IndexWord> labels, TransferLearningHelper transferHelper) {
        super(images, labels);
        this.transferHelper = transferHelper;
    }

    @Override
    protected Optional<DataSet> fetchImage(String url) {
        // featurized arrays are much smaller than image ones, thus faster to load, so check them first
        final Optional<INDArray> featurizedCached = featurizedCache.get(url);
        if (featurizedCached.isPresent()) {
            System.out.println("Found featurized in cache: " + url);
            return featurizedCached.map(data -> toDataSet(url, data));
        }

        return super.fetchImage(url)
                .map(dataSet -> featurize(dataSet, url));
    }

    private DataSet featurize(DataSet input, String url) {
        DataSet featurized = transferHelper.featurize(input);
        featurizedCache.put(url, featurized.getFeatures());
        return featurized;
    }

    @Override
    protected INDArray createInputMatrix(int numRows) {
        return Nd4j.create(numRows, transferHelper.unfrozenGraph().layerInputSize(0));
    }
}
