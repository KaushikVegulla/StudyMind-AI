export default function CitationCard({ citation }) {
  return (
    <div className="citation">
      <strong>{citation.document}</strong> - page {citation.page}, chunk {citation.chunk_index}
      <p>{citation.text}...</p>
    </div>
  );
}
