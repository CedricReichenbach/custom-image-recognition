package info.magnolia.ai;

import static java.util.stream.Collectors.toSet;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import org.apache.xerces.util.URI;

import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.IndexWord;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.Synset;
import net.sf.extjwnl.dictionary.Dictionary;

public class ImageIndex {

    private final List<String> availableSynsets;
    /** Limit samples per label to reduce imbalance (and reduce training time) */
    private final int MAX_IMAGES_PER_LABEL = 500;

    /**
     * Mapping from url to labels.
     */
    private final Map<String, Set<IndexWord>> images = new HashMap<>();

    private final List<IndexWord> labels;

    public ImageIndex() {
        availableSynsets = fetchLines("http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list");
        try {
            List<IndexWord> requestedLabels = loadLabels();
            labels = loadImageInfo(requestedLabels);
        } catch (IOException | JWNLException | URISyntaxException e) {
            throw new RuntimeException("Failed to load label list", e);
        }
    }

    private List<IndexWord> loadLabels() throws IOException, JWNLException, URISyntaxException {
        Path path = Paths.get(getClass().getResource("labels.txt").toURI());
        List<String> labelStrings = Files.readAllLines(path, Charset.forName("utf-8"));

        Dictionary dictionary = Dictionary.getDefaultResourceInstance();
        List<IndexWord> labels = new ArrayList<>();
        for (String labelString : labelStrings)
            labels.add(dictionary.lookupIndexWord(POS.NOUN, labelString));
        return labels;
    }

    private List<IndexWord> loadImageInfo(List<IndexWord> requestedLabels) {
        List<IndexWord> supportedLabels = new ArrayList<>();
        for (IndexWord label : requestedLabels) {
            try {
                loadForLabel(label);
                supportedLabels.add(label);
            } catch (NoSupportedSynsetException e) {
                System.out.println("Skipping word because no supported synset: " + label);
            }
        }
        return supportedLabels;
    }

    private void loadForLabel(IndexWord label) throws NoSupportedSynsetException {
        System.out.println("Loading image URLs: " + label.getLemma());

        final List<Synset> senses = label.getSenses();
        // XXX: Necessary to trigger lazy loading, currently not done on stream() due to: https://github.com/extjwnl/extjwnl/issues/25
        senses.iterator();
        Set<String> synsetIds = senses.stream()
                .map(synset -> String.format("n%08d", synset.getOffset()))
                .filter(availableSynsets::contains)
                .collect(toSet());
        if (synsetIds.isEmpty())
            throw new NoSupportedSynsetException("No supported synsets found for label: " + label.getLemma());

        Set<String> urls = new HashSet<>();
        for (String synsetId : synsetIds) {
            List<String> lines = fetchLines("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synsetId);

            if (lines.size() == 1 && !URI.isWellFormedAddress(lines.get(0)))
                throw new IllegalStateException(String.format("Fetching URLs for '%s' caused problems: '%s'", synsetId, lines.get(0)));
            if (lines.size() < 100)
                System.out.println(String.format("WARNING: '%s' only has %s images", synsetId, lines.size()));

            urls.addAll(lines);
        }

        urls = limitRandomized(urls);

        for (String url : urls) {
            if (!images.containsKey(url)) images.put(url, new HashSet<>());
            images.get(url).add(label);
        }
    }

    /**
     * Pick n items given by limit. Items are picked pseudo-randomly but reproducibly (based on hash).
     */
    private Set<String> limitRandomized(Set<String> items) {
        if (items.size() <= MAX_IMAGES_PER_LABEL) return items;

        List<String> list = new ArrayList<>(items);
        list.sort(Comparator.comparingInt(String::hashCode));
        return new HashSet<>(list.subList(0, MAX_IMAGES_PER_LABEL));
    }

    private List<String> fetchLines(String url) {
        List<String> lines = new ArrayList<>();
        try (Scanner scanner = new Scanner(new URL(url).openStream())) {
            while (scanner.hasNextLine()) {
                final String line = scanner.nextLine();
                if (!line.isEmpty()) lines.add(line);
            }
            return lines;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public List<IndexWord> getLabels() {
        return new ArrayList<>(labels);
    }

    public Map<String, Set<IndexWord>> getImages() {
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
}
