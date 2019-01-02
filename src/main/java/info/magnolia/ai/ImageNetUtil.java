package info.magnolia.ai;

import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.Synset;
import net.sf.extjwnl.dictionary.Dictionary;

public class ImageNetUtil {

    private static Dictionary dictionary = null;
    static {
        try {
            dictionary = Dictionary.getDefaultResourceInstance();
        } catch (JWNLException e) {
            throw new RuntimeException(e);
        }
    }

    public static String toImageNetId(Synset synset) {
        return String.format("n%08d", synset.getOffset());
    }

    public static Synset fromImageNetId(String synsetId) {
        long index = Long.parseLong(synsetId.substring(1));
        try {
            return dictionary.getSynsetAt(POS.NOUN, index);
        } catch (JWNLException e) {
            throw new RuntimeException(e);
        }
    }
}
