'use strict';

export class ApiErrors { 

    static readonly ERRORS: {[key: string] : string} = {
        
        'not-found': 'A value is missing or incorrect',
        'string-error': 'Text expected',
        'short-string-error': 'Text expected',
        'long-string-error': 'Text expected',
        'address-error': 'The address is incorrect',
        'number-error': 'Number expected',
        'float-error': 'Float expected',
        'integer-error': 'A value is missing or incorrect',
        'boolean-error': 'Boolean expected',
        'object-error': 'Object expected',
        'percent-error': 'Percentage expected',
        'integer-json-array-error': 'Integer array expected',
        'hex-error': 'Hex string expected',
        'hex-color-error': 'Hex color code expected',
        'sha256-error': 'SHA256 expected',
        'phone-error': 'Phone is invalid',
        'email-error': 'Email is invalid',
        'password-too-short': 'Password is too short',
        'password-error': 'Bad password',
        'true-expected': 'True expected',
        'date-error': 'Date expected',
        'already-exists': 'Value already exists',

        'unknown-credentials': 'Unknown credentials (login and/or password)',

        'user-already-exists': 'User already exists',

        'person-assignment-error-2': 'The end date must be later than the start date',
        'person-assignment-error-3': 'The occupation percentage must be greater than 0',
        'person-assignment-error-4': 'The cumulative occupation of a person over a given period cannot exceed 100%',

        'wbs-assignment-error-2': 'The end date must be later than the start date',
        'wbs-assignment-error-3': 'The assignment percentage must be greater than 0',
        'wbs-assignment-error-4': 'The cumulative WBS assignments of a position over a given period cannot exceed 100%',

        'company-assignment-error-1': 'The end date must be later than the start date',
        'company-assignment-error-2': 'Overlaps detected with other company assignment',

        'file-too-large': 'The file is to large',
        'file-error': 'File error',
        'file-mime-error': 'Bad file type',
        'file-security-check-fails': 'File security check fails'
    }

    /*
    **
    **
    */
    static getMessage(error: string) : string | null {

        const message = ApiErrors.ERRORS[error];

        if (!message)
            return null;

        return message;
    }
}