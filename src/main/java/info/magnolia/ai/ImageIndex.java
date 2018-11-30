package info.magnolia.ai;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.IndexWord;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.Synset;
import net.sf.extjwnl.dictionary.Dictionary;

public class ImageIndex {

    /** Mapping from url to labels. */
    private final Map<String, Set<IndexWord>> imageLabels = new HashMap<>();

    public ImageIndex() {
        loadImageInfo();
    }

    private void loadImageInfo() throws IOException, JWNLException {
        Dictionary dictionary = Dictionary.getDefaultResourceInstance();
        List<String> labels = Files.readAllLines(Paths.get("labels.txt"), Charset.forName("utf-8"));
        for (String label : labels) {
            IndexWord word = dictionary.lookupIndexWord(POS.NOUN, label);
            dictionary.
                    loadForLabel(word);
        }
    }

    private void loadForLabel(IndexWord label) {
        final Synset synset = label.getSenses().get(0);
        String response = fetch("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset);
        String[] urls = response.split("\n");
        for (String url : urls) {
            if (!imageLabels.containsKey(url)) imageLabels.put(url, new HashSet<>());
            imageLabels.get(url).add(label);
        }
    }

    private String fetch(String s) {
        // TODO
    }
}
