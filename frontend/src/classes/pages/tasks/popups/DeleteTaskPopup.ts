'use strict';

import {
    Popup,
    Api
} from '@src/classes';

export class DeleteTaskPopup extends Popup {

    constructor(private id: number) {

        super({
            validText: 'Delete',
            validRed: true,
            cancellable: true,
            title: `Delete task confirmation`,
            message: `Do you want to delete task ${id}?`
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

            await Api.post('/task/delete', {
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