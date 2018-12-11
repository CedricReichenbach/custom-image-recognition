package info.magnolia.ai;

import static java.util.stream.Collectors.toList;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import net.sf.extjwnl.data.IndexWord;

public class ImageNetDataFetcher extends BaseDataFetcher {

    private final NativeImageLoader imageLoader = new NativeImageLoader(224, 224, 3);
    private final VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
    private final FileSystemCache cache = new FileSystemCache();

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
        List<String> toFetch = urls.subList(cursor, Math.min(cursor + numExamples, urls.size()));
        List<DataSet> dataSets = toFetch.stream()
                .map(this::fetchImage)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(toList());
        if (!dataSets.isEmpty())
            this.initializeCurrFromList(dataSets);

        cursor += numExamples;
    }

    private Optional<DataSet> fetchImage(String url) {
        Optional<INDArray> cached = cache.get(url);
        if (cached.isPresent()) {
            System.out.println("Loaded image from cache: " + url);
            return cached.map(arr -> toDataSet(url, arr));
        }

        try {
            BufferedImage image = ImageIO.read(new URL(url));
            if (image == null) throw new IOException("Failed to read image from url: " + url);

            INDArray matrix = imageLoader.asMatrix(image);
            preProcessor.transform(matrix);

            System.out.println("Successfully fetched image: " + url);

            cache.put(url, matrix);
            return Optional.of(toDataSet(url, matrix));
        } catch (IOException e) {
            System.out.println("Skipping image; failed to fetch: " + url);
            return Optional.empty();
        }
    }

    private DataSet toDataSet(String url, INDArray matrix) {
        return new DataSet(matrix, oneHotEncode(images.get(url)));
    }

    private INDArray oneHotEncode(Set<IndexWord> indexWords) {
        float[] array = new float[labels.size()];
        indexWords.forEach(word -> array[labels.indexOf(word)] = 1);
        return new NDArray(array);
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
