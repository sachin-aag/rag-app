from llama_index.core import StorageContext, load_index_from_storage
import sys
import os

def test_index_metadata():
    """Test if the index has URL metadata properly applied"""
    try:
        # Get the root directory (parent of tests folder)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load the index from the root directory
        index_path = os.path.join(root_dir, "index_storage")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        # Get all nodes from the index
        nodes = index.docstore.docs.values()
        
        print(f"\nTesting metadata for {len(nodes)} documents...")
        print("-" * 50)
        
        success = True
        missing_url_docs = []
        mismatched_url_docs = []
        
        for i, node in enumerate(nodes, 1):
            metadata = node.metadata
            
            # Check if required fields exist
            has_filename = 'file_name' in metadata
            has_url = 'url' in metadata
            
            # Get the values if they exist
            filename = metadata.get('file_name', 'MISSING')
            url = metadata.get('url', 'MISSING')
            
            # Print status for each document
            print(f"\nDocument {i}:")
            print(f"Filename: {filename}")
            print(f"URL: {url}")
            print(f"Has filename: {'✓' if has_filename else '✗'}")
            print(f"Has URL: {'✓' if has_url else '✗'}")
            
            # Track documents with issues
            if not has_url:
                missing_url_docs.append(filename)
                success = False
            elif has_filename and has_url:
                expected_url = filename.split('/')[-1].replace('.md', '')\
                                    .replace('www_', 'www.')\
                                    .replace('_com_', '.com/')\
                                    .replace('_', '-')
                if url != expected_url:
                    print(f"⚠️ URL format mismatch!")
                    print(f"Expected: {expected_url}")
                    print(f"Got: {url}")
                    mismatched_url_docs.append((filename, url, expected_url))
                    success = False
        
        print("\n" + "=" * 50)
        if success:
            print("✅ All documents have proper metadata!")
        else:
            print("❌ Some documents have metadata issues:")
            if missing_url_docs:
                print("\nDocuments missing URL metadata:")
                for doc in missing_url_docs:
                    print(f"- {doc}")
            
            if mismatched_url_docs:
                print("\nDocuments with incorrect URL format:")
                for doc, got_url, expected_url in mismatched_url_docs:
                    print(f"- {doc}")
                    print(f"  Expected: {expected_url}")
                    print(f"  Got: {got_url}")
            
            print("\nPlease rebuild the index using index_data_llamaind.py")
        
        return success
            
    except Exception as e:
        print(f"\n❌ Error loading or checking index: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_index_metadata()
    sys.exit(0 if success else 1) 