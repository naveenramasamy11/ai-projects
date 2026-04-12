"""
Project 13 — LangChain Document QA
====================================
Load a text (or PDF) document, split it into chunks, embed into a FAISS
vector store, and answer natural-language questions using a RetrievalQA chain.

This is the foundational Retrieval-Augmented Generation (RAG) pattern:
  Load → Split → Embed → Retrieve → Generate

Usage:
    export OPENAI_API_KEY="your-key-here"
    python langchain_document_qa.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Sample document — AWS Cloud Services overview
# The script writes this to sample_document.txt before loading it.
# Replace the path in main() with your own .txt or .pdf file.
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT_TEXT = """
Amazon Web Services (AWS) is the world's most comprehensive and broadly adopted cloud platform,
offering over 200 fully featured services from data centres globally. AWS is used by millions
of customers — from fast-growing start-ups to the world's largest enterprises and leading
government agencies — to lower costs, become more agile, and innovate faster.

== Compute ==
Amazon EC2 (Elastic Compute Cloud) provides scalable virtual servers in the cloud. You can
choose from a wide selection of instance types optimised for different use cases, including
compute-optimised (C-series), memory-optimised (R-series), storage-optimised (I-series),
and GPU instances (P and G series). EC2 Auto Scaling adjusts capacity automatically to
maintain steady, predictable performance at the lowest possible cost.

AWS Lambda is a serverless compute service that lets you run code without provisioning or
managing servers. Lambda executes code only when triggered, scaling automatically from a few
requests per day to thousands per second. You pay only for the compute time consumed.

Amazon ECS (Elastic Container Service) and Amazon EKS (Elastic Kubernetes Service) manage
containerised workloads. ECS is the AWS-native container orchestrator, while EKS provides
managed Kubernetes for teams already invested in the Kubernetes ecosystem. AWS Fargate can
be used with both ECS and EKS to run containers without managing the underlying EC2 instances.

== Storage ==
Amazon S3 (Simple Storage Service) is object storage built to store and retrieve any amount
of data from anywhere. S3 delivers 99.999999999% (eleven 9s) of durability and stores data
for millions of applications. S3 storage classes include S3 Standard, S3 Intelligent-Tiering,
S3 Standard-IA (Infrequent Access), S3 Glacier Instant Retrieval, and S3 Glacier Deep Archive.

Amazon EBS (Elastic Block Store) provides persistent block storage volumes for use with EC2
instances. EBS volumes are network-attached and persist independently from the life of an
instance. EBS offers SSD-backed (gp3, io2) and HDD-backed (st1, sc1) volume types.

Amazon EFS (Elastic File System) provides a simple, scalable, elastic NFS file system for use
with AWS Cloud services and on-premises resources. EFS automatically grows and shrinks as you
add and remove files, with no need for capacity management.

== Databases ==
Amazon RDS (Relational Database Service) makes it easy to set up, operate, and scale a
relational database in the cloud. RDS supports Amazon Aurora, PostgreSQL, MySQL, MariaDB,
Oracle Database, and SQL Server. Multi-AZ deployments provide high availability.

Amazon DynamoDB is a fully managed, serverless, key-value NoSQL database designed to run
high-performance applications at any scale. DynamoDB offers single-digit millisecond latency
at any scale and supports both key-value and document data models.

Amazon Redshift is a fully managed, petabyte-scale data warehouse service in the cloud. It
allows you to query structured and semi-structured data across your data warehouse, operational
database, and data lake using standard SQL.

== Networking ==
Amazon VPC (Virtual Private Cloud) lets you provision a logically isolated section of the AWS
Cloud where you can launch AWS resources in a virtual network that you define. You have complete
control over your virtual networking environment, including selection of your own IP address
range, creation of subnets, and configuration of route tables and network gateways.

Amazon CloudFront is a fast content delivery network (CDN) service that securely delivers data,
videos, applications, and APIs to customers globally with low latency and high transfer speeds.
CloudFront integrates with AWS Shield for DDoS protection and AWS WAF for application-level
security.

AWS Direct Connect establishes a dedicated network connection from your premises to AWS,
bypassing the public internet for consistent, low-latency connectivity.

== Security & Identity ==
AWS IAM (Identity and Access Management) enables you to manage access to AWS services and
resources securely. Using IAM, you can create and manage AWS users and groups, and use
permissions to allow and deny their access to AWS resources. IAM roles are the recommended
way to grant permissions to EC2 instances, Lambda functions, and other AWS services.

AWS KMS (Key Management Service) makes it easy to create and manage cryptographic keys and
control their use across a wide range of AWS services and in your applications.

Amazon GuardDuty is a threat detection service that continuously monitors for malicious
activity and unauthorised behaviour to protect your AWS accounts, workloads, and data.

== Infrastructure as Code ==
AWS CloudFormation provides a common language for you to describe and provision all the
infrastructure resources in your cloud environment in a safe, repeatable way. Templates can
be written in JSON or YAML.

AWS CDK (Cloud Development Kit) is an open-source software development framework to define
your cloud application resources using familiar programming languages such as TypeScript,
Python, Java, and C#. CDK synthesises into CloudFormation templates under the hood.

HashiCorp Terraform is widely used alongside AWS to define, provision, and manage AWS
infrastructure using a declarative configuration language (HCL). Terraform state files track
the current state of infrastructure, enabling incremental updates and drift detection.
Terraform modules allow infrastructure patterns to be packaged and reused across environments.

Ansible is commonly used for AWS configuration management and application deployment,
integrating with the AWS SDK via the community.aws collection to provision and configure
resources post-provisioning. Packer can be used with Ansible provisioners to build
hardened AWS AMIs as immutable infrastructure artefacts.

== Migration ==
AWS Migration Hub provides a single place to discover your existing servers, plan migrations,
and track the status of each application migration. It integrates with AWS Application
Migration Service (MGN) for lift-and-shift server migrations.

AWS DMS (Database Migration Service) helps migrate databases to AWS quickly and securely.
The source database remains fully operational during the migration, minimising downtime.
DMS supports homogeneous migrations (e.g., Oracle to Oracle on RDS) and heterogeneous
migrations (e.g., SQL Server to Aurora PostgreSQL).
"""


def create_sample_document(path: str) -> None:
    """Write the embedded sample document to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOCUMENT_TEXT.strip())
    print(f"  Sample document written to '{path}' ({len(SAMPLE_DOCUMENT_TEXT)} chars)")


# ---------------------------------------------------------------------------
# Step 1: Load and split the document
# ---------------------------------------------------------------------------
def load_and_split(doc_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Load a .txt file with TextLoader then split into overlapping chunks.

    For PDF files, swap TextLoader for:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(doc_path)

    The RecursiveCharacterTextSplitter tries to split on paragraph breaks first
    (\n\n), then line breaks (\n), then sentences, then words — preserving as
    much semantic coherence as possible within each chunk.
    """
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    print(f"  Loaded {len(documents)} document(s) from '{doc_path}'")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Step 2: Embed chunks into a FAISS vector store
# ---------------------------------------------------------------------------
def build_vectorstore(chunks):
    """
    Convert each chunk into a dense embedding vector using OpenAI's
    text-embedding-3-small model, then index them in FAISS for fast
    similarity search at query time.
    """
    print("  Generating OpenAI embeddings — this may take a few seconds...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  FAISS index built with {vectorstore.index.ntotal} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Step 3: Build the RetrievalQA chain
# ---------------------------------------------------------------------------
def build_qa_chain(vectorstore):
    """
    Wire a retriever + LLM into a RetrievalQA chain.

    chain_type="stuff" means all retrieved chunks are stuffed into a single
    prompt. Works well when k is small (3-5 chunks). For very long documents
    use chain_type="map_reduce" or chain_type="refine" instead.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # fetch top-3 most semantically similar chunks
    )

    prompt_template = """You are a helpful assistant that answers questions based strictly
on the provided context. If the answer is not contained in the context, respond with
"I don't know based on the provided document." Do not fabricate information.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,   # lets us inspect which chunks were used
        chain_type_kwargs={"prompt": prompt},
    )
    print("  RetrievalQA chain ready")
    return qa_chain


# ---------------------------------------------------------------------------
# Step 4: Ask a question
# ---------------------------------------------------------------------------
def ask(chain, question: str) -> str:
    """Run a question through the QA chain and print a formatted answer."""
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"   └─ Based on {len(sources)} retrieved chunk(s)")
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Validate API key upfront
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "Run: export OPENAI_API_KEY='your-key-here'"
        )

    doc_path = "sample_document.txt"

    print("\n══ Step 1: Preparing document ══════════════════════════════")
    create_sample_document(doc_path)

    print("\n══ Step 2: Loading and splitting ═══════════════════════════")
    chunks = load_and_split(doc_path)

    print("\n══ Step 3: Building FAISS vector store ═════════════════════")
    vectorstore = build_vectorstore(chunks)

    print("\n══ Step 4: Building RetrievalQA chain ══════════════════════")
    qa_chain = build_qa_chain(vectorstore)

    print("\n══ Step 5: Answering preset questions ══════════════════════")
    preset_questions = [
        "What is Amazon EC2 and what instance types are available?",
        "How durable is Amazon S3 and what storage classes does it offer?",
        "What is the difference between ECS and EKS?",
        "How does Terraform integrate with AWS?",
        "What AWS services are used for database migration?",
    ]
    for q in preset_questions:
        ask(qa_chain, q)

    print(f"\n{'═'*60}")
    print("Interactive Mode — type a question or 'quit' to exit")
    print(f"{'═'*60}")
    while True:
        user_input = input("\nYour question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        ask(qa_chain, user_input)


if __name__ == "__main__":
    main()
