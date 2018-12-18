package info.magnolia.ai.cache;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ArrayCache extends FileSystemCache<INDArray> {
    public ArrayCache(String dirName) {
        super(dirName);
    }

    protected Optional<INDArray> readFromFile(File file) throws IOException {
        if (file.length() == 0) return Optional.of(Nd4j.empty());

        try (InputStream stream = new FileInputStream(file)) {
            return Optional.of(Nd4j.read(stream));
        }
    }

    protected void writeToFile(INDArray array, File file) throws IOException {
        if (array.isEmpty())
            // signal empty array simply by empty file
            file.createNewFile();
        else
            Nd4j.saveBinary(array, file);
    }
}
