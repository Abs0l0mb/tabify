'use strict';

import {
    Popup,
    Api
} from '@src/classes';

export class DeleteDocumentPopup extends Popup {

    constructor(private id: number) {

        super({
            validText: 'Delete',
            validRed: true,
            cancellable: true,
            title: `Delete document confirmation`,
            message: `Do you want to delete document ${id}?`
        });

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/document/delete', {
                id: this.id
            });

            this.hide();
            
            this.emit('done');
            
        } catch(error: any) {

            console.log(error);
            
            this.unlock();
            this.validButton.unload();
        }
    }
}