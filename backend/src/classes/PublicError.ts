'use strict';

export class PublicError extends Error {

    public readonly public: true = true;

    constructor(message?: string) {

        super(message);

        this.name = 'PublicError';
        
        Object.setPrototypeOf(this, new.target.prototype);
    }
}