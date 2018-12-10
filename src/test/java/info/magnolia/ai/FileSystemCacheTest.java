package info.magnolia.ai;

import static org.junit.Assert.*;

import java.util.Optional;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FileSystemCacheTest {

    private FileSystemCache cache;

    @Before
    public void setUp() {
        cache = new FileSystemCache();
    }

    @After
    public void tearDown() {
        cache.clear();
    }

    @Test
    public void shouldCache() {
        INDArray array = Nd4j.create(new float[]{1, 2, 3});

        assertFalse(cache.isCached("foo"));
        assertEquals(Optional.empty(), cache.get("foo"));

        cache.put("foo", array);

        assertTrue(cache.isCached("foo"));
        assertEquals(array, cache.get("foo").get());
    }
}