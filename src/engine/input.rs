#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingInputKind {
    Query,
    Document,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingInput<'a> {
    pub text: &'a str,
    pub kind: EmbeddingInputKind,
}

impl<'a> EmbeddingInput<'a> {
    pub fn query(text: &'a str) -> Self {
        Self {
            text,
            kind: EmbeddingInputKind::Query,
        }
    }

    pub fn document(text: &'a str) -> Self {
        Self {
            text,
            kind: EmbeddingInputKind::Document,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_query_constructor() {
        assert_eq!(
            EmbeddingInput::query("hello"),
            EmbeddingInput {
                text: "hello",
                kind: EmbeddingInputKind::Query,
            }
        );
    }

    #[test]
    fn test_embedding_input_document_constructor() {
        assert_eq!(
            EmbeddingInput::document("hello"),
            EmbeddingInput {
                text: "hello",
                kind: EmbeddingInputKind::Document,
            }
        );
    }
}
