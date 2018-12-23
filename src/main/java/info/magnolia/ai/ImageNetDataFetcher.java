package info.magnolia.ai;

import static java.util.stream.Collectors.toList;

import info.magnolia.ai.cache.ArrayCache;
import info.magnolia.ai.cache.FileSystemCache;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.sf.extjwnl.data.IndexWord;

public class ImageNetDataFetcher extends BaseDataFetcher {

    private static final Logger log = LoggerFactory.getLogger(BaseDataFetcher.class);

    private static final long FETCH_TIMEOUT = 1000;

    private final ExecutorService fetcherPool = Executors.newCachedThreadPool();

    private final NativeImageLoader imageLoader = new NativeImageLoader(224, 224, 3);
    private final VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
    private final FileSystemCache cache = new ArrayCache("custom-image-recognition-samples");

    private final Map<String, Set<IndexWord>> images;
    private final List<String> urls;
    private final List<IndexWord> labels;

    public ImageNetDataFetcher(Map<String, Set<IndexWord>> images, List<IndexWord> labels) {
        this.totalExamples = images.keySet().size();
        this.numOutcomes = labels.size();

        this.images = images;
        this.urls = new ArrayList<>(images.keySet());
        this.labels = labels;
    }

    @Override
    public void fetch(int numExamples) {
        log.info("Going to fetch up to {} sample images, starting at {}...", numExamples, cursor);

        List<String> toFetch = urls.subList(cursor, Math.min(cursor + numExamples, urls.size()));
        List<DataSet> dataSets = toFetch.parallelStream()
                .map(this::fetchImage)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(toList());
        if (!dataSets.isEmpty())
            this.initializeCurrFromList(dataSets);

        cursor += numExamples;

        log.info("Batch done");
    }

    protected Optional<DataSet> fetchImage(String url) {
        Optional<INDArray> cached = cache.get(url);
        if (cached.isPresent()) {
            if (Nd4j.empty().equals(cached.get())) {
                log.debug("Skipping image (previous failure signaled by cache): {}", url);
                return Optional.empty();
            }

            log.debug("Loaded image from cache: ", url);
            return cached.map(arr -> toDataSet(url, arr));
        }

        try {
            BufferedImage image = readImage(url);

            INDArray matrix = imageLoader.asMatrix(image);
            preProcessor.transform(matrix);

            log.debug("Successfully fetched image: ", url);

            // Commented out because only caching not-found ones currently (as found ones are cached in featurized form anyway)
            // cache.put(url, matrix);

            return Optional.of(toDataSet(url, matrix));
        } catch (Exception e) { // might not just be IOException but e.g. IllegalStateException in case of invalid encoding (server might return 200 with HTML)
            log.debug("Skipping image; failed to fetch: ", url);
            // cache empty matrix signaling missing data
            cache.put(url, Nd4j.empty());
            return Optional.empty();
        }
    }

    private BufferedImage readImage(String url) throws IOException {
        try {
            BufferedImage image = fetcherPool.submit(() -> ImageIO.read(new URL(url)))
                    .get(FETCH_TIMEOUT, TimeUnit.MILLISECONDS);
            if (image == null) throw new IOException("Failed to read image from url: " + url);
            return image;
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            log.debug("Image-fetching interrupted", e);
            throw new IOException("Fetching image was interrupted: " + url, e);
        }
    }

    protected DataSet toDataSet(String url, INDArray matrix) {
        return new DataSet(matrix, oneHotEncode(images.get(url)));
    }

    private INDArray oneHotEncode(Set<IndexWord> indexWords) {
        float[] array = new float[labels.size()];
        // FIXME: How does labels sometimes not contain one of the words?
        indexWords.forEach(word -> array[labels.indexOf(word)] = 1);
        return Nd4j.create(array);
    }

    @Override
    protected INDArray createInputMatrix(int numRows) {
        return Nd4j.create(numRows, 3, 224, 224);
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException("This shape is higher-dimensional");
    }
}
