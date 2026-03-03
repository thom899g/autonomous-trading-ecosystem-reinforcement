"""
Firebase Client for Autonomous Trading Ecosystem
Handles all state management, real-time data streaming, and persistent storage
Critical for ecosystem resilience and distributed state management
"""
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, db
from firebase_admin.exceptions import FirebaseError

# Type hints
from google.cloud.firestore import Client as FirestoreClient

class FirebaseClient:
    """Singleton Firebase client with automatic reconnection and state recovery"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Firebase client with error handling and auto-recovery
        
        Args:
            config_path: Path to Firebase service account JSON (optional, uses environment by default)
        
        Raises:
            ValueError: If Firebase initialization fails
            FirebaseError: For Firebase-specific errors
        """
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._firestore_client: Optional[FirestoreClient] = None
        self._realtime_db = None
        
        try:
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                if config_path and os.path.exists(config_path):
                    cred = credentials.Certificate(config_path)
                elif os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
                    # Load from environment variable
                    import json
                    service_account_info = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
                    cred = credentials.Certificate(service_account_info)
                else:
                    # Use default application credentials (for GCP)
                    cred = credentials.ApplicationDefault()
                
                # Initialize Firebase
                firebase_admin.initialize_app(cred, {
                    'projectId': os.getenv('FIREBASE_PROJECT_ID', 'autonomous-trading-ecosystem'),
                    'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
                })
            
            # Initialize clients
            self._firestore_client = firestore.client()
            self._realtime_db = db.reference('/')
            
            self._initialized = True
            self.logger.info("✅ Firebase client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Firebase initialization failed: {str(e)}")
            raise ValueError(f"Firebase initialization failed: {str(e)}")
    
    def save_trading_state(self, 
                          agent_id: str, 
                          state: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save trading agent state to Firestore with transaction safety
        
        Args:
            agent_id: Unique identifier for trading agent
            state: Current trading state dictionary
            metadata: Optional metadata (timestamp, version, etc.)
        
        Returns:
            bool: Success status
        
        Raises:
            FirebaseError: If Firestore operation fails
        """
        if not self._initialized:
            self.logger.error("Firebase client not initialized")
            return False
        
        try:
            # Create document with metadata
            document_data = {
                'state': state,
                'metadata': metadata or {},
                'timestamp': firestore.SERVER_TIMESTAMP,
                'agent_id': agent_id,
                'version': state.get('version', '1.0')
            }
            
            # Use transaction for atomic write
            transaction = self._firestore_client.transaction()
            doc_ref = self._firestore_client.collection('trading_states').document(agent_id)
            
            @firestore.transactional
            def update_in_transaction(transaction, doc_ref, new_data):
                transaction.set(doc_ref, new_data)
            
            update_in_transaction(transaction, doc_ref, document_data)
            
            self.logger.debug(f"✅ Trading state saved for agent {agent_id}")