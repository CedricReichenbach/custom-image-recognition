package info.magnolia.ai.cache;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;

public class LinesCache extends FileSystemCache<List<String>> {

    public LinesCache(String dirName) {
        super(dirName);
    }

    @Override
    protected Optional<List<String>> readFromFile(File file) throws IOException {
        return Optional.of(Files.readAllLines(file.toPath()));
    }

    @Override
    protected void writeToFile(List<String> lines, File file) throws IOException {
        Files.write(file.toPath(), lines);
    }
}
