package info.magnolia.ai.cache;

import static org.junit.Assert.*;

import java.util.Optional;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ArrayCacheTest {

    private ArrayCache cache;

    @Before
    public void setUp() {
        cache = new ArrayCache("file-system-array-cache-test");
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

    @Test
    public void shouldCacheEmptyArray() {
        assertFalse(cache.isCached("empty"));
        assertEquals(Optional.empty(), cache.get("empty"));

        cache.put("empty", Nd4j.empty());

        assertTrue(cache.isCached("empty"));
        assertEquals(Nd4j.empty(), cache.get("empty").get());
    }
}