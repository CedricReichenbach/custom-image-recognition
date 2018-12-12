package info.magnolia.ai;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FileSystemCache {

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

    public Optional<INDArray> get(String key) {
        File file = getTempFile(key);

        if (!file.exists()) return Optional.empty();
        if (file.length() == 0) return Optional.of(Nd4j.empty());

        try (InputStream stream = new FileInputStream(file)) {
            return Optional.of(Nd4j.read(stream));
        } catch (IOException e) {
            e.printStackTrace();
            return Optional.empty();
        }
    }

    public void put(String key, INDArray array) {
        try {
            if (array.isEmpty())
                // signal empty array simply by empty file
                getTempFile(key).createNewFile();
            else
                Nd4j.saveBinary(array, getTempFile(key));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

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
