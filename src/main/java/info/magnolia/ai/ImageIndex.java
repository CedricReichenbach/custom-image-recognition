package info.magnolia.ai;

import info.magnolia.ai.cache.LinesCache;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.sf.extjwnl.data.Synset;
import org.yaml.snakeyaml.Yaml;

public class ImageIndex {

    private static final Logger log = LoggerFactory.getLogger(ImageIndex.class);

    private final List<String> availableSynsets;
    /**
     * Limit samples per label to reduce imbalance (and reduce training time)
     */
    private final int MAX_IMAGES_PER_LABEL = 120;
    private final int MIN_IMAGES_PER_LABEL = 1000;

    /**
     * Mapping from url to labels.
     */
    private final Map<String, Set<Synset>> images = new ConcurrentHashMap<>();

    private final List<Synset> labels;

    private final LinesCache urlsCache = new LinesCache("imagenet-urls");

    public ImageIndex() {
        availableSynsets = fetchLines("http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list");
        List<Synset> requestedLabels = loadLabels();
        labels = loadImageInfo(requestedLabels);
    }

    private List<Synset> loadLabels() {
        try (InputStream stream = getClass().getResourceAsStream("labels-synsets-popular.yaml")) {
            Map<String, String> labelMap = new Yaml().load(stream);

            List<Synset> labels = new ArrayList<>();
            for (String labelString : labelMap.keySet()) {
                Synset synset = ImageNetUtil.fromImageNetId(labelString);
                if (synset == null) throw new IllegalStateException("Synset unknown: " + labelString);
                labels.add(synset);
            }
            return labels;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private List<Synset> loadImageInfo(List<Synset> requestedLabels) {
        List<Synset> supportedLabels = new ArrayList<>();
        requestedLabels.parallelStream()
                .forEach(label -> {
                    try {
                        loadForLabel(label);
                        supportedLabels.add(label);
                    } catch (NoSupportedSynsetException e) {
                        log.warn("Skipping word because no supported synset: {}", label);
                    } catch (NotEnoughSamplesException e) {
                        log.warn("Skipping word because not enough sample images: {} ({})", label, e.getMessage());
                    }
                });
        supportedLabels.sort(Comparator.comparing(Synset::getIndex));
        return supportedLabels;
    }

    private void loadForLabel(Synset label) throws NoSupportedSynsetException, NotEnoughSamplesException {
        log.info("Loading image URLs: {} ({})", label.getIndex(), label.getGloss());

        String synsetId = ImageNetUtil.toImageNetId(label);
        if (!availableSynsets.contains(synsetId))
            throw new NoSupportedSynsetException("Synset not supported: " + label.getOffset());

        List<String> urls = fetchLines("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synsetId);

        if (urls.size() == 1 && !urls.get(0).startsWith("http")) {
            log.error("Fetching URLs for '{}' caused problems: '{}'", synsetId, urls.get(0));
        }

        if (urls.size() < MIN_IMAGES_PER_LABEL)
            throw new NotEnoughSamplesException(String.format("Only %s sample image(s)", urls.size()));

        urls = limitRandomized(urls);

        for (String url : urls) {
            if (!images.containsKey(url)) images.put(url, Collections.synchronizedSet(new HashSet<>()));
            images.get(url).add(label);
        }
    }

    /**
     * Pick n items given by limit. Items are picked pseudo-randomly but reproducibly (based on hash).
     */
    private List<String> limitRandomized(List<String> items) {
        if (items.size() <= MAX_IMAGES_PER_LABEL) return items;

        List<String> list = new ArrayList<>(items);
        list.sort(Comparator.comparingInt(String::hashCode));
        return list.subList(0, MAX_IMAGES_PER_LABEL);
    }

    private List<String> fetchLines(String url) {
        Optional<List<String>> cached = urlsCache.get(url);
        if (cached.isPresent()) return cached.get();

        List<String> lines = new ArrayList<>();
        try (Scanner scanner = new Scanner(new URL(url).openStream())) {
            while (scanner.hasNextLine()) {
                final String line = scanner.nextLine();
                if (!line.isEmpty()) lines.add(line);
            }
            urlsCache.put(url, lines);
            return lines;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public List<Synset> getLabels() {
        return new ArrayList<>(labels);
    }

    public Map<String, Set<Synset>> getImages() {
        return images;
    }

    @Override
    public String toString() {
        return String.format("ImageIndex (%s labels, %s image URLs)", labels.size(), images.keySet().size());
    }

    private static class NoSupportedSynsetException extends Exception {
        NoSupportedSynsetException(String message) {
            super(message);
        }
    }

    private static class NotEnoughSamplesException extends Exception {
        NotEnoughSamplesException(String message) {
            super(message);
        }
    }
}
