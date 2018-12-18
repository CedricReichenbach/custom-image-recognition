package info.magnolia.ai.cache;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class FileSystemCache<T> {

    private static final Logger log = LoggerFactory.getLogger(FileSystemCache.class);

    private static final String SUFFIX = ".cache";
    private static final String MGNL_FOLDER = ".mgnl";
    private final File dir;

    public FileSystemCache(String dirName) {
        dir = new File(System.getProperty("user.home") + File.separator + MGNL_FOLDER, dirName);
        dir.mkdirs();
    }

    public boolean isCached(String key) {
        return get(key).isPresent();
    }

    public Optional<T> get(String key) {
        File file = getTempFile(key);

        if (!file.exists()) return Optional.empty();

        try {
            return readFromFile(file);
        } catch (IOException e) {
            log.error("Failed to read from cache file", e);
            return Optional.empty();
        }
    }

    protected abstract Optional<T> readFromFile(File file) throws IOException;

    public void put(String key, T item) {
        try {
            writeToFile(item, getTempFile(key));
        } catch (IOException e) {
            log.error("Failed to write to cache file", e);
        }
    }

    protected abstract void writeToFile(T item, File file) throws IOException;

    private File getTempFile(String key) {
        String escaped = key.replaceAll("[^\\w-]+", "_");
        return new File(dir, escaped + SUFFIX);
    }

    public void clear() {
        try {
            Files.walk(dir.toPath())
                    .map(Path::toFile)
                    .sorted((a, b) -> -a.compareTo(b))
                    .forEach(File::delete);
        } catch (IOException e) {
            throw new RuntimeException("Failed to delete image cache temp dir", e);
        }
    }
}
