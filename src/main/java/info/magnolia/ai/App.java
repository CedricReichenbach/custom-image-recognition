package info.magnolia.ai;

public class App {
    public static void main(String[] args) {
        ImageIndex imageIndex = new ImageIndex();
        System.out.println(imageIndex);

        NetworkManager manager = new NetworkManager(imageIndex.getLabels());
        // TODO
    }
}
